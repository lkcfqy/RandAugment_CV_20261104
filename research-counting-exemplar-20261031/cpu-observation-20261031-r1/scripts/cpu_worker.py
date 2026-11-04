"""Fixed Mac CPU repeatability or separately authorized full observation.

Derived from the audited worker. No GPU or cross-device numerical claim.
"""
import os,sys,time,json,hashlib,socket,traceback,threading,resource,gc
from pathlib import Path
from evidence import Evidence,atomic_bytes
from contracts import canonical,compute_gate,complete_result_set,validate_query,bound_metric_inputs
from model_io import file_hash

ROOT=Path(__file__).resolve().parents[1]
START=time.monotonic()
MEMORY_LIMIT=8*1024**3

def main():
    import argparse,importlib.metadata as metadata
    parser=argparse.ArgumentParser();parser.add_argument('--context',type=Path,required=True);args=parser.parse_args()
    context_file=json.loads(args.context.read_text());mode=context_file['mode'];output=Path(context_file['output'])
    if context_file['supervisor_pid']!=os.getppid():raise RuntimeError('external supervisor required')
    plan=json.loads((ROOT/'plan.json').read_text());evidence=Evidence(output,owned_root=ROOT.parent)
    for package,version in plan['runtime_versions'].items():
        if metadata.version(package)!=version:raise RuntimeError('frozen CPU environment version differs: '+package)
    for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(output/'cache/matplotlib'),'HF_HOME':str(output/'cache/hf'),'CUDA_VISIBLE_DEVICES':'','PYTORCH_ENABLE_MPS_FALLBACK':'0'}.items():os.environ[key]=value
    sys.dont_write_bytecode=True
    phase='precheck' if mode=='precheck' else 'formal';model=None;hooks=None;monitor=None;stop=threading.Event();calls=[];results=[]
    counters={'root_forward_reserved':0,'root_forward_entered':0,'root_forward_returned':0,'precheck_attempts':0,'formal_attempts':0,'formal_completed':0,'network_attempts':0,'pickle_attempts':0,'science_reads_during_precheck':0,'training_steps':0,'GPU_operation_attempts':0}
    memory={'peak_rss_bytes':0}
    report={'status':'RUNNING','previous_compute_wall_seconds':context_file['previous_compute_wall_seconds'],'freeze_sha256':context_file['freeze_sha256'],'mode':mode,'backend':'Mac CPU FP32 pinned HF/app basic path','counters':counters,'precheck_calls':calls,'formal_execution_started':False,'fixture_changes':0,'retries':0}
    def compute_used():return 7200.-(context_file['compute_deadline_monotonic']-time.monotonic())
    def save_status():
        atomic_bytes(output/'worker_state.json',canonical({'phase':phase,'compute_used_seconds':compute_used(),'memory':memory,'counters':counters}))
    def checkpoint():
        report['compute_used_seconds']=compute_used();report['worker_wall_seconds']=time.monotonic()-START
        evidence.write('engineering/status.json',report);save_status()
    def forbidden_network(*args,**kwargs):
        counters['network_attempts']+=1;raise RuntimeError('network forbidden after asset preparation')
    socket.socket.connect=forbidden_network;socket.socket.connect_ex=forbidden_network;socket.create_connection=forbidden_network
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).absolute()
            if phase=='precheck' and (ROOT/'science'==p or ROOT/'science' in p.parents):
                counters['science_reads_during_precheck']+=1;raise RuntimeError('science data forbidden during synthetic precheck')
    sys.addaudithook(audit)
    checkpoint()
    try:
        import random,numpy as np,torch
        from PIL import Image
        from model_io import load_fixed_model,registered_snapshot
        from hooks import NativeHooks,token_contract,tensor_hash
        from numerics import validate,compare,NumericalFailure
        from fixture import POSITIVE,NEGATIVE,TEXTS,WIDTH,HEIGHT
        from metrics import summarize
        def reject_pickle(*args,**kwargs):counters['pickle_attempts']+=1;raise RuntimeError('pickle checkpoint forbidden')
        torch.load=reject_pickle
        random.seed(42);np.random.seed(42);torch.random.default_generator.manual_seed(42)
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        report['CPU_resources']={'os_cpu_count':os.cpu_count(),'affinity_cpu_ids':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC')},'cgroup_cpu_limits':{str(p):p.read_text().strip() for p in (Path('/sys/fs/cgroup/cpu.max'),Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')) if p.is_file()},'policy':'fixed 2 intra-op / 1 inter-op; no resource-driven tuning'}
        checkpoint()
        def gpu_guard(api):
            # Distinct function identities preserve TorchDynamo's per-API rule registry.
            def forbidden(*args,**kwargs):
                counters['GPU_operation_attempts']+=1;raise RuntimeError('GPU operation forbidden: '+api)
            return forbidden
        torch.Tensor.cuda=gpu_guard('Tensor.cuda');torch.nn.Module.cuda=gpu_guard('Module.cuda')
        for namespace,names in [(torch.cuda,('_lazy_init','init','synchronize','empty_cache','set_device')),(torch.mps,('synchronize','empty_cache','current_allocated_memory','driver_allocated_memory'))]:
            for name in names:
                if hasattr(namespace,name):setattr(namespace,name,gpu_guard(namespace.__name__+'.'+name))
        report['runtime']={'python':sys.version,'versions':{k:metadata.version(k) for k in plan['runtime_versions']},'device':'cpu','dtype':'FP32','AMP':False,'seed':42,'CPU_default_generator_seeded_without_GPU_seed_calls':True}
        model,count,aliases,loading=load_fixed_model(ROOT);report['strict_loading']=loading
        model_cpu_snapshot=registered_snapshot(model,aliases)
        if any(p.device.type!='cpu' for p in model.parameters()):raise RuntimeError('non-CPU parameter')
        report['loading_and_snapshot_wall_seconds']=time.monotonic()-START
        ctx={};hooks=NativeHooks(model,ctx,counters)
        fixture_manifest=json.loads((ROOT/'fixture/manifest.json').read_text())
        if fixture_manifest['positive_xyxy']!=POSITIVE or fixture_manifest['negative_xyxy']!=NEGATIVE or fixture_manifest['texts']!={k:list(v) for k,v in TEXTS.items()}:raise RuntimeError('frozen fixture contract differs')
        for concept,(pos,neg) in TEXTS.items():token_contract(model,pos,neg)
        memory_lock=threading.Lock()
        def sample_memory():
            with memory_lock:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform=='linux' else 1)
                memory['peak_rss_bytes']=max(memory['peak_rss_bytes'],rss)
                save_status()
        def memory_thread():
            while not stop.is_set():
                try:sample_memory()
                except Exception:os._exit(72)
                if memory['peak_rss_bytes']>MEMORY_LIMIT:os._exit(73)
                stop.wait(.1)
        monitor=threading.Thread(target=memory_thread,daemon=True);monitor.start()
        references={};reference_inputs={};reference_topk={};reference_hashes={}
        def reserve(label):
            if compute_used()>=7140:raise RuntimeError('worker computation limit reached; 60 s reserved for finalization/verification')
            if phase=='precheck':
                if counters['precheck_attempts']>=6:raise RuntimeError('precheck attempt cap reached')
                counters['precheck_attempts']+=1
            else:
                if counters['formal_attempts']>=492:raise RuntimeError('formal attempt cap reached')
                counters['formal_attempts']+=1
            counters['root_forward_reserved']+=1
            evidence.write('attempts/%04d.json'%counters['root_forward_reserved'],{'index':counters['root_forward_reserved'],'label':label,'phase':phase,'status':'STARTED','compute_used_seconds':compute_used()})
            save_status()
        def execute(image,positive,negative,text,neg_text,device,mode):
            hooks.reset();ctx.update(phase=phase,mode=mode,device=device,positive=positive,negative=negative,original_wh=list(image.size),caption=text+' . '+neg_text+' . ',expected_ids=token_contract(model,text,neg_text))
            with torch.no_grad(),torch.autocast(device_type=device,enabled=False):
                result=count(model,image,text,exemplar_boxes=positive,negative_texts=[neg_text],negative_exemplar_boxes=[negative],conf_thresh=.23)
            validity=validate(hooks.raw)
            actual_mask=hooks.raw['native_filter_mask']
            if int(actual_mask.sum())!=result['count']:raise RuntimeError('native device mask disagrees with returned count')
            if not torch.equal(hooks.raw['pred_boxes'][0][actual_mask],torch.from_numpy(result['boxes'])):raise RuntimeError('native returned boxes disagree with native mask')
            plain={'count':result['count'],'boxes':result['boxes'].tolist(),'scores':result['scores'].tolist()}
            canonical(plain)  # serialization cost included; synthetic values are never persisted.
            return plain,validity
        def synthetic(round_number,mode_name):
            begin=time.monotonic();label='CPU_'+mode_name+'_repeat_'+str(round_number);reserve(label)
            raw=(ROOT/'fixture/texture.rgb').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=fixture_manifest['pixel_sha256']:raise RuntimeError('fixed synthetic pixels changed')
            image=Image.frombytes('RGB',(WIDTH,HEIGHT),raw)
            result,validity=execute(image,POSITIVE,NEGATIVE,*TEXTS[mode_name],'cpu',mode_name)
            del result,image,raw
            record={'mode':mode_name,'device':'cpu','repeat':round_number,'shape':ctx['actual_shape'],'validity':validity,'input_hashes':dict(hooks.inputs),'complete_query_seconds':time.monotonic()-begin}
            if round_number==1:
                references[mode_name]=dict(hooks.raw);reference_inputs[mode_name]=dict(hooks.inputs);reference_topk[mode_name]=dict(hooks.topk)
                reference_hashes[mode_name]={k:tensor_hash(v) for k,v in hooks.raw.items()}
            else:
                if hooks.inputs!=reference_inputs[mode_name]:raise RuntimeError('CPU repeat input/token/box hashes differ')
                record['comparison_scope']='same CPU/runtime/input, first vs second; no CUDA comparison'
                record['numerical_comparison']=compare(references[mode_name],hooks.raw)
                record['actual_native_mask_equal']=torch.equal(references[mode_name]['native_filter_mask'],hooks.raw['native_filter_mask'])
                if not record['actual_native_mask_equal']:raise RuntimeError('CPU repeat native masks differ')
                record['reference_clone_unchanged']={k:tensor_hash(v) for k,v in references[mode_name].items()}==reference_hashes[mode_name]
                if not record['reference_clone_unchanged']:raise RuntimeError('CPU reference clone changed')
                record['topk_order_equal']=torch.equal(reference_topk[mode_name]['indices'],hooks.topk['indices'])
                record['raw_exact_equal']=all(torch.equal(references[mode_name][key],hooks.raw[key]) for key in references[mode_name])
            calls.append(record)
            evidence.write('attempts/%04d.json'%counters['root_forward_reserved'],{'index':counters['root_forward_reserved'],'label':label,'phase':phase,'status':'COMPLETE','compute_used_seconds':compute_used()})
            checkpoint()
            record['complete_query_seconds']=time.monotonic()-begin
            record['timing_includes']='fixture hash/decode, token and input assertions, CPU forward, output validation/comparison, serialization and first durable checkpoint; final annotation covered by 60 s reserve'
            checkpoint()
        if mode=='precheck':
            for round_number in (1,2):
                for mode_name in ('F','C','V'):synthetic(round_number,mode_name)
            report['post_repeat_registered_state_exact_equal']=registered_snapshot(model,aliases)==model_cpu_snapshot
            if not report['post_repeat_registered_state_exact_equal']:raise RuntimeError('model state changed across CPU repeats')
            timings=[max(x['complete_query_seconds'] for x in calls if x['mode']==m) for m in ('F','C','V')]
            report['planned_full_492_gate']=compute_gate(compute_used(),30.,timings,60.)
            report['formal_startup_allowance_seconds']=30.;report['CPU_repeatability_is_not_CUDA_equivalence_or_correctness']=True
            report['status']='PASS_CPU_REPEATABILITY';phase='complete'
            return 0
        approved_precheck=json.loads(Path(context_file['precheck_engineering']).read_text())
        if approved_precheck['status']!='PASS_CPU_REPEATABILITY' or approved_precheck['freeze_sha256']!=context_file['freeze_sha256']:raise RuntimeError('matching passed CPU precheck required')
        timings=[max(x['complete_query_seconds'] for x in approved_precheck['precheck_calls'] if x['mode']==m) for m in ('F','C','V')]
        gate=compute_gate(compute_used(),0.,timings,60.)
        report['compute_gate']=gate;checkpoint()
        if not gate['allowed']:raise RuntimeError('original 2x full492 forecast exceeds cumulative 7200 s budget')
        report['formal_execution_started']=True;phase='formal';save_status()
        science=ROOT/'science';manifest=json.loads((ROOT/'manifests/science_files.json').read_text())
        for rel,sha in manifest.items():
            if file_hash(science/rel)!=sha:raise RuntimeError('frozen science asset hash mismatch: '+rel)
        selection=json.loads((science/'selection_frozen.json').read_text());labels=json.loads((science/'labels_frozen.json').read_text());query_document=json.loads((science/'queries_frozen.json').read_text());protocol=json.loads((ROOT/'execution_contract.json').read_text())
        if file_hash(ROOT/'execution_contract.json')!=plan['execution_contract_sha256'] or file_hash(ROOT/'scripts/metrics.py')!=protocol['metric_code_sha256']:raise RuntimeError('current execution protocol/metric identity mismatch')
        queries=query_document['queries'];source_map={x['filename']:x for x in selection['images']};label_map={x['pair_id']:x for x in labels['pairs']}
        if len(queries)!=492 or len(source_map)!=94 or len(label_map)!=47 or sum(x['eligible'] for x in labels['pairs'])!=41:raise RuntimeError('frozen denominator/count changed')
        for name,value in [('selection',selection),('labels',labels),('queries',query_document),('protocol',protocol)]:evidence.write('inputs/'+name+'.json',value)
        report['input_manifest_hashes']={name:file_hash(science/name) for name in ['selection_frozen.json','labels_frozen.json','queries_frozen.json']}
        report['input_manifest_hashes']['actual_execution_contract']=file_hash(ROOT/'execution_contract.json')
        evidence.write('inputs/historical_protocol_v2_NOT_CURRENT.json',json.loads((science/'protocol_v2_frozen.json').read_text()))
        checkpoint()
        for index,q in enumerate(queries):
            reserve(q['query_id']);begin=time.monotonic();source=source_map[q['filename']];label=label_map[q['pair_id']]
            validate_query(q,source,label)
            if Path(q['filename']).name!=q['filename']:raise RuntimeError('unsafe image filename')
            image_path=science/'images'/q['filename'];image_bytes=image_path.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest()!=q['image_sha256']:raise RuntimeError('per-query original image hash changed')
            import io
            image=Image.open(io.BytesIO(image_bytes)).convert('RGB')
            for box in q['exemplar_boxes']+q['negative_exemplar_boxes'][0]:
                x1,y1,x2,y2=box
                if not (0<=x1<x2<=image.width and 0<=y1<y2<=image.height):raise RuntimeError('frozen exemplar out of image')
            result,validity=execute(image,q['exemplar_boxes'],q['negative_exemplar_boxes'][0],q['text'],q['negative_texts'][0],'cpu',q['mode'])
            row={'query_id':q['query_id'],'query_sha256':q['query_sha256'],'pair_id':q['pair_id'],'filename':q['filename'],'mode':q['mode'],'target':q['target'],'input_contract':q,'original_annotation_identity':source,'prediction':result,'box_format':'normalized_cx_cy_w_h','actual_shape':ctx['actual_shape'],'original_wh':list(image.size),'saturated_ge_800':result['count']>=800,'complete_query_seconds_before_persist':time.monotonic()-begin,'compute_used_seconds':compute_used(),'validity':validity,'method':'CountGD++ pinned HF/app basic path on fixed Mac CPU runtime; confidence .23; no adaptive crop/pseudo-exemplars; no CUDA equivalence claim'}
            evidence.write('results/%04d.json'%index,row);results.append(row);counters['formal_completed']+=1
            evidence.write('attempts/%04d.json'%counters['root_forward_reserved'],{'index':counters['root_forward_reserved'],'label':q['query_id'],'phase':phase,'status':'COMPLETE','result_path':'results/%04d.json'%index,'compute_used_seconds':compute_used()})
            del result,image,image_bytes
            save_status()
        complete_result_set(results,{q['query_id']:q['query_sha256'] for q in queries})
        grouped={}
        for row in results:grouped.setdefault(row['filename'],{}).setdefault(row['mode'],{})[row['target']]=row['prediction']['count']
        paired,policy=bound_metric_inputs(grouped,labels['pairs'])
        summary=summarize(paired,policy)
        summary.update({'completed_queries':492,'expected_queries':492,'complete':True,'prediction_box_review':'PENDING; numeric gate alone does not support correct-target-selection explanation','saturated_query_ids':[x['query_id'] for x in results if x['saturated_ge_800']]})
        evidence.write('science/summary.json',summary)
        report['status']='COMPLETE_FIXED_OBSERVATION_PENDING_BOX_INTERPRETATION';phase='complete'
    except Exception as error:
        report['status']='ENGINEERING_BLOCKED' if not report['formal_execution_started'] else 'INCOMPLETE_FORMAL_OBSERVATION'
        report['error_type']=type(error).__name__;report['error']=('CPU repeat numerical validation failed' if type(error).__name__=='NumericalFailure' else str(error));report['traceback']=traceback.format_exc()
        if hasattr(error,'report'):report['numerical_failure_diagnostics']=error.report
        report['missing_queries']=492-counters['formal_completed']
        phase='failed'
    finally:
        stop.set()
        if monitor is not None:monitor.join(timeout=1)
        if hooks is not None:hooks.close()
        model=None;gc.collect()
        report['memory']=memory;checkpoint();evidence.write('completion.json',{'status':report['status'],'formal_completed':counters['formal_completed'],'expected_queries':492,'complete':report['status']=='PASS_CPU_REPEATABILITY' if mode=='precheck' else counters['formal_completed']==492 and report['status'].startswith('COMPLETE'), 'mode':mode,'compute_used_seconds':compute_used()});evidence.finish_index()
    return 0 if report['status'].startswith('COMPLETE') or report['status']=='PASS_CPU_REPEATABILITY' else 2

if __name__=='__main__':raise SystemExit(main())
