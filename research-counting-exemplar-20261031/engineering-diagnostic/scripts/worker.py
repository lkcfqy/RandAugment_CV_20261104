"""One separately authorized CPU F / MPS F diagnostic. No dataset access."""
import os
import sys
from pathlib import Path
import time
import json
import hashlib
import resource
import socket
import threading
import traceback
import random
import gc

from controls import AttemptLedger, GuardViolation, save_engineering_json

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parent
SOURCE = ROOT/'sources/countgdpp-hf'
REPORTS = PHASE/'run'
START = time.monotonic()
LIMIT = 8*1024**3


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    launch = json.loads((PHASE/'launch_once.json').read_text())
    if launch['supervisor_pid'] != os.getppid() or launch['worker_sha256'] != digest(__file__):
        raise GuardViolation('worker requires the claimed external supervisor')
    for key, value in {'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'HF_HUB_DISABLE_TELEMETRY':'1',
                       'CUDA_VISIBLE_DEVICES':'', 'PYTORCH_ENABLE_MPS_FALLBACK':'0', 'PYTHONDONTWRITEBYTECODE':'1',
                       'TOKENIZERS_PARALLELISM':'false', 'MPLCONFIGDIR':str(REPORTS/'mplconfig'),
                       'HF_HOME':str(REPORTS/'hf-cache')}.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    counters = {'forward_attempts_reserved':0, 'top_model_entered':0, 'top_model_returned':0,
                'training_steps':0, 'torch_load_attempts':0, 'socket_connect_attempts':0,
                'dataset_read_attempts':0, 'dataset_files_read':0, 'formal_queries':0}
    report = {'status':'RUNNING', 'counters':counters, 'calls':[], 'events':[],
              'wall_limit_seconds':120, 'attempt_limit':2, 'rss_limit_bytes':LIMIT, 'mps_limit_bytes':LIMIT,
              'seed':42, 'real_utility_measured':False, 'prediction_payloads_persisted':False,
              'padding_policy_fixed_before_forward':'native masked positions must be exactly -inf; active logits and all coordinates finite; sigmoid scores finite',
              'absolute_tolerance':1e-4, 'relative_tolerance_against_cpu':1e-3,
              'confidence_threshold':.23, 'input_original_size_wh':[1333,800], 'input_expected_shape':[1,3,800,1333],
              'cpu_intraop_threads':2, 'cpu_interop_threads':1, 'fallback_enabled':False,
              'source_modifications':0, 'retries':0}
    def save():
        report['elapsed_seconds'] = time.monotonic()-START
        save_engineering_json(REPORTS/'engineering.json', report)
    def event(name, **kw):
        row = {'event':name, 'elapsed_seconds':time.monotonic()-START, **kw}
        report['events'].append(row)
        save(); print(json.dumps(row), flush=True)
    def block_network(*args, **kwargs):
        counters['socket_connect_attempts'] += 1
        raise GuardViolation('network forbidden during synthetic precheck')
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    def audit(event_name, args):
        if event_name != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        forbidden = [ROOT/'data', ROOT/'sources/pairtally', ROOT/'reports/audit']
        is_forbidden = any(p == x or x in p.parents for x in forbidden)
        if ROOT/'reports' in p.parents and p != ROOT/'reports/state_manifest.json':
            is_forbidden = True
        if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
            is_forbidden = True
        if is_forbidden:
            counters['dataset_read_attempts'] += 1
            raise GuardViolation('dataset/image file access forbidden in engineering precheck')
    sys.addaudithook(audit)
    event('worker_started')
    model = None; torch = None; monitor = None; stop = threading.Event(); mps_used = False
    memory = {'peak_rss_bytes':0, 'mps_current_bytes':0, 'mps_driver_bytes':0, 'peak_mps_current_bytes':0, 'peak_mps_driver_bytes':0}
    handles = []
    ledger = AttemptLedger(REPORTS/'attempts.json', maximum=2)
    diagnostic = None
    raw_reference_hashes = {}
    try:
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from PIL import Image
        from safetensors.torch import load_file
        from numerics import validate, compare, NumericalFailure
        from instrumentation import Recorder, snapshot, tensor_hash
        def fail_pickle(*args, **kwargs):
            counters['torch_load_attempts'] += 1
            raise GuardViolation('torch.load and pickle checkpoints forbidden')
        torch.load = fail_pickle
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        report['mps_available'] = torch.backends.mps.is_available()
        if not report['mps_available']:
            raise GuardViolation('MPS is unavailable; no CPU-only substitution')
        event('runtime_imported', torch_version=torch.__version__)
        source_before = {str(p.relative_to(SOURCE)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
        prior = json.loads((ROOT/'cpu-readiness/reports/cpu_verification.json').read_text())
        assert source_before == prior['source_sha256_before'], 'audited native source identity changed'
        assert digest(ROOT/'models/model.safetensors') == '94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5'
        assert digest(ROOT/'models/bert-base-uncased/model.safetensors') == '68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3'
        sys.path.insert(0, str(SOURCE))
        from countgdpp import CountGDPlusPlus, count
        from models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
        from models.GroundingDINO import ms_deform_attn
        from models.GroundingDINO import groundingdino_app as grounding_module
        assert ms_deform_attn._C is None, 'unexpected custom CUDA extension'
        load_start = time.monotonic()
        config = json.loads((SOURCE/'config.json').read_text())
        config['text_encoder_type'] = str(ROOT/'models/bert-base-uncased')
        config['device'] = 'cpu'
        assert 'auto_map' not in json.loads((ROOT/'models/bert-base-uncased/config.json').read_text())
        model = CountGDPlusPlus(config=config).eval().requires_grad_(False)
        manifest = json.loads((ROOT/'reports/state_manifest.json').read_text())
        expected = manifest['expected_schema']; aliases = manifest['expected_shared_aliases']
        physical = load_file(str(ROOT/'models/model.safetensors'), device='cpu')
        state = model.state_dict()
        assert set(physical) == set(expected)
        assert set(state) == set(expected)|set(aliases)
        registered = dict(model.named_parameters(remove_duplicate=False))
        registered.update(dict(model.named_buffers(remove_duplicate=False)))
        nonpersistent = {name+'.'+key for name,module in model.named_modules() for key in module._non_persistent_buffers_set if module._buffers.get(key) is not None}
        assert set(registered)-set(state) == nonpersistent == {'bert.embeddings.position_ids','bert.embeddings.token_type_ids'}
        assert torch.equal(registered['bert.embeddings.position_ids'], torch.arange(512).expand(1,-1))
        assert torch.equal(registered['bert.embeddings.token_type_ids'], torch.zeros((1,512),dtype=torch.long))
        assert not set(state)-set(registered)
        def signature(t):
            return t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device
        for alias, canonical in aliases.items():
            assert alias not in physical and canonical in physical
            assert registered[alias] is registered[canonical]
            assert signature(state[alias]) == signature(state[canonical])
            assert manifest['tensor_metadata']['__metadata__'][alias] == canonical
        payload = dict(physical)
        payload.update({alias:physical[canonical] for alias,canonical in aliases.items()})
        assert set(payload) == set(state)
        for key,tensor in payload.items():
            assert tensor.shape == state[key].shape and tensor.dtype == state[key].dtype
            assert tensor.device.type == state[key].device.type == 'cpu'
        for tensor in physical.values(): assert bool(torch.isfinite(tensor).all())
        result = model.load_state_dict(payload, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.equal(t, physical[aliases.get(key,key)]) for key,t in model.state_dict().items())
        assert all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules())
        report['strict_loading'] = {'logical_keys':len(state), 'physical_keys':len(physical), 'actual_shared_aliases':len(aliases),
                                    'strict':True, 'missing_keys':[], 'unexpected_keys':[], 'all_exactly_equal':True}
        del payload, physical, state, registered, manifest, tensor
        gc.collect()
        report['cpu_load_seconds'] = time.monotonic()-load_start
        event('strict_cpu_load_complete', load_seconds=report['cpu_load_seconds'])

        # Predeclare three token contracts using actual native helpers with artificial embeddings only.
        modes = {'F':('red toy','blue toy')}
        expected_ids = {}
        for mode,(positive,negative) in modes.items():
            tokenized = model.tokenizer([positive+' . '+negative+' . '], padding='longest', return_tensors='pt')
            attention, positions, _ = generate_masks_with_special_tokens_and_transfer_map(tokenized, model.specical_tokens, None)
            td = {'encoded_text':torch.zeros(1,tokenized['input_ids'].shape[1],256), 'text_token_mask':tokenized['attention_mask'].bool(), 'position_ids':positions, 'text_self_attention_masks':attention}
            td,tokenized = model.add_exemplar_tokens(tokenized,td,torch.full((1,3,256),11.),[torch.tensor([0])])
            td,tokenized = model.add_exemplar_tokens(tokenized,td,torch.full((1,3,256),22.),[torch.tensor([1])])
            ids = tokenized['input_ids'][0].tolist(); first = ids.index(1012); second = ids.index(1012,first+1)
            positive_slots = [i for i in range(first) if ids[i] == 1008]
            negative_slots = [i for i in range(first+1,second) if ids[i] == 1008]
            assert len(positive_slots) == len(negative_slots) == 3 and ids.count(1008) == 6
            assert bool((td['encoded_text'][0,positive_slots] == 11).all()) and bool((td['encoded_text'][0,negative_slots] == 22).all())
            assert not td['text_self_attention_masks'][0,positive_slots][:,negative_slots].any()
            assert not td['text_self_attention_masks'][0,negative_slots][:,positive_slots].any()
            if mode == 'V': assert ids == [101,1008,1008,1008,1012,1008,1008,1008,1012,102]
            expected_ids[mode] = ids
        report['native_token_ids_expected'] = expected_ids
        event('F_helper_contract_passed')
        image = Image.new('RGB',(1333,800),(100,110,120))
        positives = [[64.,64.,96.,96.],[160.,64.,192.,96.],[256.,64.,288.,96.]]
        negatives = [[64.,192.,96.,224.],[160.,192.,192.,224.],[256.,192.,288.,224.]]
        report['synthetic_input'] = {'rgb':[100,110,120], 'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(), 'positive_xyxy':positives, 'negative_xyxy':negatives, 'text_modes':modes}
        context = {}; captured = {}
        report['diagnostic_only'] = True
        report['previous_compute_wall_seconds'] = 34.21403079200536
        report['precision_settings'] = {'environment':{key:os.environ.get(key,'UNSET') for key in ['PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL','PYTORCH_ENABLE_MPS_FALLBACK','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG']}, 'float32_matmul_precision':torch.get_float32_matmul_precision(), 'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(), 'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32, 'mps_fp32_precision_override':'none; unchanged installed native defaults', 'mps_name':torch.backends.mps.get_name(), 'mps_core_count':torch.backends.mps.get_core_count()}
        diagnostic = Recorder(model,grounding_module,context,report,save)
        def migration_state():
            tensors=dict(model.named_parameters(remove_duplicate=False));tensors.update(dict(model.named_buffers(remove_duplicate=False)))
            hashes={key:tensor_hash(value) for key,value in tensors.items()}
            shared={alias:canonical for alias,canonical in aliases.items() if tensors[alias] is tensors[canonical] and signature(tensors[alias])==signature(tensors[canonical])}
            assert shared==aliases
            return {'tensor_sha256':hashes,'actual_shared_aliases':shared,'keys':len(hashes),'device':next(model.parameters()).device.type}
        migration_cpu=migration_state()
        report['migration_cpu_snapshot']=migration_cpu
        diagnostic.install()
        def before_forward(module,args,kwargs):
            counters['top_model_entered'] += 1
            assert counters['top_model_entered'] == counters['forward_attempts_reserved'] <= 2
            assert len(args[3]) == len(args[4]) == 1 and len(args[4][0]) == 1
            for nested in (args[0],args[1],args[3][0]):
                assert tuple(nested.tensors.shape) == (1,3,800,1333)
                assert nested.tensors.dtype == torch.float32 and nested.tensors.device.type == context['device']
            assert torch.equal(args[2][0].cpu(),torch.tensor(positives))
            assert torch.equal(args[4][0][0].cpu(),torch.tensor(negatives))
            assert kwargs['captions'] == [modes[context['mode']][0]+' . '+modes[context['mode']][1]+' . ']
            for name,nested in [('query',args[0]),('positive',args[1]),('negative',args[3][0])]:
                diagnostic.exact_input('normalized_input.'+name,nested.tensors)
                diagnostic.exact_input('input_padding_mask.'+name,nested.mask)
            diagnostic.exact_input('positive_xyxy',args[2][0])
            diagnostic.exact_input('negative_xyxy',args[4][0][0])
            assert all(p.device.type == context['device'] for p in module.parameters())
            save()
        def after_forward(module,args,output):
            counters['top_model_returned'] += 1
            captured.clear()
            for key in ['pred_logits','pred_boxes','input_ids','text_mask']:
                captured[key] = snapshot(output[key])
            assert captured['input_ids'][0].tolist() == expected_ids[context['mode']]
            assert captured['pred_logits'].dtype == captured['pred_boxes'].dtype == torch.float32
            diagnostic.exact_input('final_native_token_ids',captured['input_ids'])
            diagnostic.exact_input('final_native_text_mask',captured['text_mask'])
        handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True))
        handles.append(model.register_forward_hook(after_forward))
        memory_lock = threading.Lock()
        def sample_memory():
            with memory_lock:
                memory['peak_rss_bytes'] = max(memory['peak_rss_bytes'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if mps_used:
                    memory['mps_current_bytes'] = torch.mps.current_allocated_memory()
                    memory['mps_driver_bytes'] = torch.mps.driver_allocated_memory()
                    memory['peak_mps_current_bytes'] = max(memory['peak_mps_current_bytes'], memory['mps_current_bytes'])
                    memory['peak_mps_driver_bytes'] = max(memory['peak_mps_driver_bytes'], memory['mps_driver_bytes'])
                save_engineering_json(REPORTS/'memory.json', memory)
        def monitor_memory():
            while not stop.is_set():
                try: sample_memory()
                except Exception as exc:
                    save_engineering_json(REPORTS/'memory_monitor_failure.json', {'exception':type(exc).__name__,'message':str(exc)})
                    os._exit(72)
                if memory['peak_rss_bytes'] > LIMIT or memory['peak_mps_driver_bytes'] > LIMIT or memory['peak_mps_current_bytes'] > LIMIT:
                    os._exit(73)
                stop.wait(.05)
        monitor = threading.Thread(target=monitor_memory, daemon=True)
        monitor.start()
        references = {}
        def run_one(device, mode, warmup=False):
            context.update(device=device,mode=mode)
            label = device.upper()+'_'+mode+('_warmup' if warmup else '')
            ledger.reserve(label)
            counters['forward_attempts_reserved'] = len(ledger.rows)
            event('attempt_started', label=label, attempt=counters['forward_attempts_reserved'])
            began = time.monotonic()
            try:
                if device == 'mps': torch.mps.synchronize()
                # Native path includes preprocessing, H2D, forward, native filtering, and D2H.
                with torch.no_grad():
                    result = count(model,image,modes[mode][0],exemplar_boxes=positives,
                                   negative_texts=[modes[mode][1]],negative_exemplar_boxes=[negatives],conf_thresh=.23)
                del result
                if device == 'mps': torch.mps.synchronize()
                elapsed = time.monotonic()-began
                validity = validate(captured)
                record = {'label':label,'mode':mode,'device':device,'warmup':warmup,'end_to_end_seconds':elapsed,
                          'input_shape':[1,3,800,1333], 'validity':validity, 'native_group_assertions_passed':True}
                if device == 'cpu':
                    references[mode] = dict(captured)
                    raw_reference_hashes.update({{'pred_logits':'output_logits','pred_boxes':'output_coordinates','input_ids':'token_ids','text_mask':'text_mask'}[key]:tensor_hash(value) for key,value in captured.items()})
                    report['cpu_result_clone_hashes_before_mps']=dict(raw_reference_hashes)
                else: record['numerical_comparison'] = compare(references[mode],captured)
                report['calls'].append(record)
                captured.clear()
                sample_memory()
                ledger.finish(True)
                event('attempt_completed',label=label,end_to_end_seconds=elapsed)
            except Exception as exc:
                ledger.finish(False,type(exc).__name__+': '+str(exc))
                if isinstance(exc,NumericalFailure): report['numerical_failure_diagnostics'] = exc.report
                report['failed_call'] = {'label':label,'elapsed_seconds_until_error':time.monotonic()-began}
                raise
        run_one('cpu','F')
        recommended = torch.mps.recommended_max_memory()
        allocator_fraction = min(1.0,LIMIT/recommended)
        torch.mps.set_per_process_memory_fraction(float(allocator_fraction))
        mps_used = True
        report['mps_allocator'] = {'recommended_bytes':recommended,'fraction':allocator_fraction,'allocator_limit_bytes':allocator_fraction*recommended,'driver_guard_bytes':LIMIT}
        move_start = time.monotonic()
        model.to('mps'); torch.mps.manual_seed(42); torch.mps.synchronize()
        report['cpu_to_mps_model_transfer_seconds'] = time.monotonic()-move_start
        migration_mps=migration_state()
        report['migration_mps_snapshot']=migration_mps
        report['migration_exact_equal']=(migration_mps['tensor_sha256']==migration_cpu['tensor_sha256'] and migration_mps['actual_shared_aliases']==migration_cpu['actual_shared_aliases'])
        assert report['migration_exact_equal'], 'CPU/MPS registered tensor values or aliases changed during migration'
        event('model_transferred_to_mps',seconds=report['cpu_to_mps_model_transfer_seconds'],all_tensor_and_alias_hashes_exact=True)
        run_one('mps','F')
        assert counters['forward_attempts_reserved'] == counters['top_model_entered'] == counters['top_model_returned'] == 2
        source_after = {str(p.relative_to(SOURCE)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
        assert source_after == source_before
        report['sources_unchanged'] = True
        report['status'] = 'PASS_SYNTHETIC_ENGINEERING_ONLY'
        report['source_sha256'] = source_before
    except Exception as exc:
        report['status'] = 'ENGINEERING_BLOCKED'
        report['exception_type'] = type(exc).__name__
        report['exception_message'] = str(exc)
        report['traceback'] = traceback.format_exc()
        event('engineering_blocked',exception_type=type(exc).__name__,message=str(exc))
    finally:
        stop.set()
        if monitor is not None: monitor.join(timeout=1)
        if diagnostic is not None:
            diagnostic.finalize()
            if raw_reference_hashes:
                after={{'pred_logits':'output_logits','pred_boxes':'output_coordinates','input_ids':'token_ids','text_mask':'text_mask'}[key]:tensor_hash(value) for key,value in references['F'].items()}
                report['cpu_result_clone_hashes_after_mps']=after
                report['cpu_result_clones_unchanged']=after==raw_reference_hashes
                assert report['cpu_result_clones_unchanged']
            diagnostic.close()
        for handle in handles: handle.remove()
        handles.clear()
        model = None
        gc.collect()
        if torch is not None and mps_used:
            try:
                torch.mps.synchronize(); torch.mps.empty_cache()
                report['mps_cleanup'] = {'current_bytes':torch.mps.current_allocated_memory(),'driver_bytes':torch.mps.driver_allocated_memory(),'cache_emptied':True}
            except Exception as cleanup_error:
                report['mps_cleanup'] = {'cache_emptied':False,'error':str(cleanup_error)}
        else:
            report['mps_cleanup'] = {'mps_never_used':True}
        report['memory'] = dict(memory)
        report['peak_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        event('worker_finished',status=report['status'],attempts=counters['forward_attempts_reserved'])
    return 0 if report['status'] == 'PASS_SYNTHETIC_ENGINEERING_ONLY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
