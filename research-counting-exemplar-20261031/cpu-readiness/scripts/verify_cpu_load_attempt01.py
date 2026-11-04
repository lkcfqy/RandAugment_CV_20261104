"""Strict CPU model loading and native prompt helpers. Never runs any model forward."""
from portable_dev.paths import project_path as _portable_project_path
import os,sys,time,json,hashlib,traceback,resource,platform,socket,collections
from pathlib import Path
ROOT=_portable_project_path(__file__, 'research-counting-exemplar-20261031');PHASE=ROOT/'cpu-readiness';SOURCE=ROOT/'sources/countgdpp-hf'
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','CUDA_VISIBLE_DEVICES':'','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false','MPLCONFIGDIR':str(PHASE/'reports/mplconfig'),'HF_HOME':str(PHASE/'reports/hf-cache')}.items():os.environ[k]=v
sys.dont_write_bytecode=True
COUNTERS={'model_forward_attempts':0,'model_forward_calls':0,'module_forward_attempts':0,'gpu_operations':0,'training_steps':0,'torch_load_attempts':0,'socket_connect_attempts':0,'recording_stub_calls':0}
START=time.time();events=[]
def save(name,value):(PHASE/'reports'/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def log(step,**kw):
    x={'step':step,'elapsed_seconds':time.time()-START,**kw};events.append(x);save('cpu_verification_events.json',events);print(json.dumps(x),flush=True)
def filehash(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fail_network(*a,**kw):COUNTERS['socket_connect_attempts']+=1;raise RuntimeError('Network forbidden during CPU verification')
socket.socket.connect=fail_network;socket.socket.connect_ex=fail_network;socket.create_connection=fail_network
import torch
from safetensors.torch import load_file
import importlib.metadata as metadata
def fail_pickle(*a,**kw):COUNTERS['torch_load_attempts']+=1;raise RuntimeError('torch.load/pickle checkpoint forbidden')
torch.load=fail_pickle
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
def forbid_module_forward(module,args):
    COUNTERS['module_forward_attempts']+=1
    if module.__class__.__name__=='CountGDPlusPlus':COUNTERS['model_forward_attempts']+=1
    raise RuntimeError('All nn.Module forwards forbidden in CPU verification: '+module.__class__.__name__)
forward_guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid_module_forward)
try:
    # Verify input asset identities before importing the now-authorized audited local source.
    assert filehash(ROOT/'models/model.safetensors')=='94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5'
    source_before={str(p.relative_to(SOURCE)):filehash(p) for p in SOURCE.rglob('*') if p.is_file()}
    sys.path.insert(0,str(SOURCE))
    from countgdpp import CountGDPlusPlus,count
    from models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
    from models.GroundingDINO import ms_deform_attn
    log('audited_source_imported',optional_cuda_extension_loaded=ms_deform_attn._C is not None)
    original_config=json.loads((SOURCE/'config.json').read_text());config=dict(original_config);config['text_encoder_type']=str(ROOT/'models/bert-base-uncased');config['device']='cpu'
    assert 'auto_map' not in json.loads((ROOT/'models/bert-base-uncased/config.json').read_text())
    save('actual_model_config.json',config)
    model=CountGDPlusPlus(config=config)
    model.eval();model.requires_grad_(False)
    log('native_model_constructed_cpu',module_count=sum(1 for _ in model.modules()))
    manifest=json.loads((ROOT/'reports/state_manifest.json').read_text());expected=manifest['expected_schema'];aliases=manifest['expected_shared_aliases']
    header=manifest['tensor_metadata'];physical=load_file(str(ROOT/'models/model.safetensors'),device='cpu')
    assert set(physical)==set(expected)
    state=model.state_dict();logical_expected=set(expected)|set(aliases)
    assert set(state)==logical_expected,{'missing':sorted(logical_expected-set(state)),'unexpected':sorted(set(state)-logical_expected)}
    registered=dict(model.named_parameters(remove_duplicate=False));registered.update(dict(model.named_buffers(remove_duplicate=False)))
    assert set(registered)==set(state)
    def storage_signature(t):return (t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),t.dtype,str(t.device))
    alias_report=[]
    for alias,canonical in aliases.items():
        assert alias not in physical and canonical in physical
        assert registered[alias] is registered[canonical],(alias,canonical,'Not the same actual registered object')
        assert storage_signature(state[alias])==storage_signature(state[canonical])
        assert header['__metadata__'][alias]==canonical
        alias_report.append({'alias':alias,'canonical':canonical,'same_registered_object':True,'same_storage_offset_shape_stride_dtype_device':True})
    expanded=dict(physical)
    for alias,canonical in aliases.items():expanded[alias]=physical[canonical]
    def validate_payload(payload):
        if set(payload)!=set(state):raise ValueError('logical key mismatch')
        for key,t in payload.items():
            target=state[key]
            if t.shape!=target.shape:raise ValueError('shape mismatch: '+key)
            if t.dtype!=target.dtype:raise ValueError('dtype mismatch: '+key)
            if t.device.type!='cpu' or target.device.type!='cpu':raise ValueError('non-CPU tensor: '+key)
    validate_payload(expanded)
    # Meaningful rejection checks do not call loader or alter any model parameter.
    rejected=[];key='feat_map.bias'
    for kind in ['missing','unexpected','shape','dtype']:
        bad=dict(expanded)
        if kind=='missing':bad.pop(key)
        elif kind=='unexpected':bad['unexpected_audit_parameter']=torch.zeros(1)
        elif kind=='shape':bad[key]=torch.zeros(257)
        else:bad[key]=bad[key].to(torch.float64)
        try:validate_payload(bad)
        except ValueError as e:rejected.append({'case':kind,'rejected':True,'reason':str(e)})
        else:raise AssertionError('Payload rejection guard failed '+kind)
    del bad
    finite_physical={}
    for key,t in physical.items():
        okay=bool(torch.isfinite(t).all().item())
        assert okay,key;finite_physical[key]=okay
    result=model.load_state_dict(expanded,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    loaded=model.state_dict();rows=[]
    for key,t in loaded.items():
        source=aliases.get(key,key);assert torch.equal(t,physical[source]),key
        assert t.device.type=='cpu' and t.dtype==expanded[key].dtype
        rows.append({'key':key,'physical_source':source,'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'equals_exact_checkpoint_tensor':True,'finite':finite_physical[source]})
    assert all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules())
    save('actual_shared_aliases.json',alias_report);save('loaded_logical_state.json',rows);save('strict_payload_rejection_checks.json',rejected)
    log('strict_cpu_load_verified',logical_keys=len(rows),physical_keys=len(physical),aliases=len(alias_report),missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,all_values_equal=True,all_finite=True)

    # Guard direct .forward calls as well as Module.__call__ for the remaining helper tests.
    def direct_forward_forbidden(*a,**kw):COUNTERS['model_forward_attempts']+=1;raise RuntimeError('Direct model forward forbidden')
    for module in model.modules():module.forward=direct_forward_forbidden
    # Exercise the ACTUAL local add_exemplar_tokens method with manual embeddings, never BERT/backbone.
    tokenizer=model.tokenizer;special=model.specical_tokens;star=tokenizer.convert_tokens_to_ids('*');dot=tokenizer.convert_tokens_to_ids('.')
    token_tests=[]
    for target in ['A','B']:
        for mode in ['F','C','V']:
            pos_text,neg_text=({'A':('red toy','blue toy'),'B':('blue toy','red toy')}[target] if mode=='F' else ('toy','') if mode=='C' else ('',''))
            caption=pos_text+' . '+neg_text+' . '
            tokenized=tokenizer([caption],padding='longest',return_tensors='pt');before=tokenized['input_ids'][0].tolist()
            mask,pos,_=generate_masks_with_special_tokens_and_transfer_map(tokenized,special,None)
            text_dict={'encoded_text':torch.zeros(1,len(before),256),'text_token_mask':tokenized['attention_mask'].bool(),'position_ids':pos,'text_self_attention_masks':mask}
            pos_value=11. if target=='A' else 22.;neg_value=22. if target=='A' else 11.
            A=torch.full((1,3,256),pos_value);B=torch.full((1,3,256),neg_value)
            text_dict,tokenized=model.add_exemplar_tokens(tokenized,text_dict,A,[torch.tensor([0])])
            text_dict,tokenized=model.add_exemplar_tokens(tokenized,text_dict,B,[torch.tensor([1])])
            after=tokenized['input_ids'][0].tolist();first=after.index(dot);second=after.index(dot,first+1)
            pos_indices=[i for i,v in enumerate(after[:first]) if v==star];neg_indices=[i for i in range(first+1,second) if after[i]==star]
            assert len(pos_indices)==len(neg_indices)==3 and after.count(star)==6
            assert torch.equal(text_dict['encoded_text'][0,pos_indices],A[0]);assert torch.equal(text_dict['encoded_text'][0,neg_indices],B[0])
            am=text_dict['text_self_attention_masks'][0]
            assert not am[pos_indices][:,neg_indices].any() and not am[neg_indices][:,pos_indices].any()
            assert am[pos_indices][:,pos_indices].all() and am[neg_indices][:,neg_indices].all()
            if mode=='V':assert after==[101,1008,1008,1008,1012,1008,1008,1008,1012,102]
            expected_tokens=tokenizer.convert_ids_to_tokens(before);after_without_stars=[i for i in after if i!=star];assert after_without_stars==before
            token_tests.append({'mode':mode,'target':target,'caption':caption,'before_ids':before,'after_ids':after,'after_tokens':tokenizer.convert_ids_to_tokens(after),'positive_visual_indices':pos_indices,'negative_visual_indices':neg_indices,'positive_sentinel':pos_value,'negative_sentinel':neg_value,'actual_native_helper_used':True,'group_attention_separated':True,'learned_forward_used':False})
    save('native_concept_token_tests.json',token_tests)
    log('native_token_helpers_verified',cases=len(token_tests))

    # Native count() preprocessing and argument contract against a NON-model recording callable.
    # Stop exactly at the learned-model invocation boundary; produce no logits/boxes/counts.
    from PIL import Image
    class StopAtModelBoundary(Exception):pass
    class Recorder:
        def __init__(self):self.record=None;self.device_token=torch.zeros(1)
        def parameters(self):return iter([self.device_token])
        def __call__(self,*args,**kw):
            COUNTERS['recording_stub_calls']+=1;self.record=(args,kw);raise StopAtModelBoundary()
    im=Image.new('RGB',(320,200),(100,110,120));boxesA=[[10.,10.,30.,30.],[40.,10.,60.,30.],[70.,10.,90.,30.]];boxesB=[[10.,60.,30.,80.],[40.,60.,60.,80.],[70.,60.,90.,80.]]
    wrapper_tests=[]
    for target in ['A','B']:
        for mode in ['F','C','V']:
            pos_text,neg_text=({'A':('red toy','blue toy'),'B':('blue toy','red toy')}[target] if mode=='F' else ('toy','') if mode=='C' else ('',''))
            posboxes,negboxes=(boxesA,boxesB) if target=='A' else (boxesB,boxesA);rec=Recorder()
            try:count(rec,im,pos_text,exemplar_boxes=posboxes,negative_texts=[neg_text],negative_exemplar_boxes=[negboxes])
            except StopAtModelBoundary:pass
            else:raise AssertionError('Expected recorder boundary')
            args,kw=rec.record
            assert kw['captions']==[pos_text+' . '+neg_text+' . ']
            assert len(args[3])==len(args[4])==1 and len(args[4][0])==1
            assert torch.equal(args[2][0],torch.tensor(posboxes)*4.) and torch.equal(args[4][0][0],torch.tensor(negboxes)*4.)
            assert tuple(args[0].tensors.shape)==(1,3,800,1280)
            assert all(t.device.type=='cpu' for t in [args[0].tensors,args[1].tensors,args[3][0].tensors,args[2][0],args[4][0][0]])
            wrapper_tests.append({'mode':mode,'target':target,'caption':kw['captions'][0],'positive_boxes_resized':args[2][0].tolist(),'negative_group_boxes_resized':args[4][0][0].tolist(),'negative_group_count':len(args[3]),'input_shape':list(args[0].tensors.shape),'actual_native_wrapper_used':True,'recording_callable_is_nn_module':False,'model_boundary_aborted_before_outputs':True})
    rec=Recorder()
    try:count(rec,im,'toy',exemplar_boxes=boxesA,negative_texts=[],negative_exemplar_boxes=[boxesB])
    except ValueError:empty_negative_list_rejected=True
    else:raise AssertionError('Empty negative text list unexpectedly accepted with one visual group')
    assert rec.record is None
    save('native_wrapper_group_tests.json',{'cases':wrapper_tests,'empty_negative_text_list_rejected_before_model_boundary':empty_negative_list_rejected,'synthetic_image_only':True,'dataset_images_read':0,'dataset_counts_read':0,'model_prediction_outputs_created':0})
    log('native_wrapper_contract_verified',cases=len(wrapper_tests),recording_stub_calls=COUNTERS['recording_stub_calls'])
    source_after={str(p.relative_to(SOURCE)):filehash(p) for p in SOURCE.rglob('*') if p.is_file()};assert source_after==source_before
    assert all(COUNTERS[k]==0 for k in COUNTERS if k!='recording_stub_calls')
    versions={d.metadata['Name']:d.version for d in metadata.distributions()};save('environment_actual_versions.json',versions)
    module_paths={n:str(sys.modules[n].__file__) for n in ['torch','torchvision','transformers','tokenizers','timm','safetensors','cv2','pycocotools','countgdpp']}
    final={'status':'PASS_CPU_LOAD_AND_NO_FORWARD_HELPERS_ONLY','counters':COUNTERS,'all_parameters_cpu':True,'all_parameters_frozen':True,'all_modules_eval':True,'strict_load_argument':True,'actual_strict_load_missing':result.missing_keys,'actual_strict_load_unexpected':result.unexpected_keys,'logical_keys':len(rows),'physical_keys':len(physical),'verified_actual_shared_aliases':len(alias_report),'all_loaded_values_exactly_equal_checkpoint':True,'all_checkpoint_values_finite':True,'native_concept_cases':len(token_tests),'native_wrapper_cases':len(wrapper_tests),'guard_rejection_cases':len(rejected),'source_sha256_before':source_before,'sources_unchanged':True,'config_changes':{'text_encoder_type':{'from':original_config['text_encoder_type'],'to':config['text_encoder_type']},'device':{'from':original_config['device'],'to':config['device']}},'architectural_code_modifications':0,'strict_false_used_by_our_loader':False,'trust_remote_code':False,'pickle_checkpoints_loaded':0,'dataset_images_used':0,'performance_or_utility_measured':False,'mps_compatibility_claimed':False,'elapsed_seconds':time.time()-START,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'platform':platform.platform(),'python':sys.version,'module_paths':module_paths,'weight_sha256':filehash(ROOT/'models/model.safetensors')}
    save('cpu_verification.json',final);log('complete',status=final['status'],counters=COUNTERS)
except Exception:
    save('cpu_verification_failure.json',{'counters':COUNTERS,'events':events,'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-START});raise
finally:
    forward_guard.remove()
