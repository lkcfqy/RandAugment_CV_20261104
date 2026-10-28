"""CPU-only source-contract and decision-path audit; no model/image training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import time,signal
START=time.perf_counter()
def limit(signum,frame):raise TimeoutError('60-second wall cap')
signal.signal(signal.SIGALRM,limit);signal.alarm(60)
import ast,collections,copy,csv,hashlib,importlib.util,json,math,resource,types
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CFG=json.loads((HERE/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvout(name,rows):
    with (HERE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def getmethod(path,cls,name):
    module=ast.parse((HERE/'source'/path).read_text())
    node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return copy.deepcopy(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name))
def compilemethod(node,scope):
    module=ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]))
    exec(compile(module,'<audited-pinned-source-method>','exec'),scope)
    return scope[node.name]
def formula(z,y):
    p=np.exp(z-z.max(axis=1,keepdims=True));p/=p.sum(1,keepdims=True)
    return (p[np.arange(len(y)),y]/np.sqrt((p*p).sum(1)))**(2/math.log(z.shape[1]))
def stats(values):
    v=np.abs(np.asarray(values,dtype=float))
    if len(v)==0:return {'n':0,'mean_abs':None,'median_abs':None,'p95_abs':None,'max_abs':None}
    return {'n':len(v),'mean_abs':float(v.mean()),'median_abs':float(np.median(v)),'p95_abs':float(np.quantile(v,.95)),'max_abs':float(v.max())}
def summary_batches(n,b,c,world=1,active=True):
    per_rank=math.ceil(n/world)
    batches=[min(b,per_rank-start) for start in range(0,per_rank,b)]
    scoring=[x-x//2 for x in batches]
    trigger=[i for i,x in enumerate(scoring) if x==c]
    return {'N':n,'classes':c,'world':world,'per_rank_sampler_size':per_rank,'sampler_padding_total':world*per_rank-n,'per_rank_full_batch':b,'loader_batch_count':len(batches),'last_loader_batch':batches[-1],'scoring_batch_histogram':dict(collections.Counter(scoring)),'triggering_batch_indices':trigger,'triggering_batches':len(trigger) if active else 0,'triggering_score_samples':len(trigger)*c if active else 0,'SRA_actually_active':active,'arithmetic_if_SRA_were_added':not active,'native_api_boundary_verified':True}

class FixedLogitEncoder:
    def __init__(self,z):self.z=z;self.calls=0
    def normalize(self,x):return x
    def net_forward(self,x,training):
        assert training and len(x)==len(self.z)
        self.calls+=1
        return self.z,torch.zeros((len(x),1),dtype=self.z.dtype)

def score_via_forward(fn,z,y):
    proxy=FixedLogitEncoder(z)
    placeholder=torch.zeros((len(z),3,32,32),dtype=torch.float32)
    output=fn(proxy,placeholder,training=True,y=y,use_basic_aug=False)
    assert proxy.calls==1 and torch.equal(output[1],z) and output[3] is placeholder
    assert output[0] is None and output[2].shape==(len(z),)
    return output[2]

# The operator order and constructor arguments are parsed from the original source.
def operator_settings():
    module=ast.parse((HERE/'source/transforms/SRA.py').read_text())
    assignment=next(n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CANDIDATE_OPS_DICT_32' for t in n.targets))
    d=assignment.value.args[0]
    return [(ast.literal_eval(k),*[ast.literal_eval(x) for x in v.args]) for k,v in zip(d.keys,d.values)]
SETTINGS=operator_settings()
ARG_FUNCTIONS={}
for name,low,high in SETTINGS:
    if low is not None:
        node=getmethod('transforms/operations.py',name,'_magnitude_to_arg')
        ARG_FUNCTIONS[name]=compilemethod(node,{'torch':torch,'np':np})

class LoggedNoPixelOp:
    def __init__(self,name,low,high):self.op_name=name;self.min_val=low;self.max_val=high;self.owner=None
    def __call__(self,x,magnitude):
        row=self.owner.current
        row['applied']=True
        row['parameter']=float(ARG_FUNCTIONS[self.op_name](self,magnitude)) if self.op_name in ARG_FUNCTIONS else None
        return x

def operation_trace(scores,seed_offset):
    torch.manual_seed(CFG['seeds']['operation_trace_torch']+seed_offset)
    np.random.seed(CFG['seeds']['operation_trace_numpy']+seed_offset)
    ops=collections.OrderedDict((name,LoggedNoPixelOp(name,low,high)) for name,low,high in SETTINGS)
    scope={'torch':torch,'np':np,'RA_SPACE':{'RA':list(ops)},'CANDIDATE_OPS_DICT_32':ops,'CANDIDATE_OPS_DICT_224':ops}
    init=compilemethod(getmethod('transforms/SRA.py','SampleAwareRandAugment','__init__'),scope)
    forward=compilemethod(getmethod('transforms/SRA.py','SampleAwareRandAugment','forward'),scope)
    original_apply=compilemethod(getmethod('transforms/SRA.py','SampleAwareRandAugment','_apply_op'),scope)
    cls=type('SampleAwareRandAugment',(torch.nn.Module,),{'__init__':init,'forward':forward})
    scope['SampleAwareRandAugment']=cls
    obj=cls(depth=2,resolution=32,augment_space='RA',p_min_t=1.0,p_max_t=1.0)
    obj.trace=[];obj.current=None
    for op in ops.values():op.owner=obj
    def logged_apply(self,x,prob,s_op,mag_mean):
        slot=len(self.trace)
        row={'sample':slot//2,'depth':slot%2,'operation':list(ops)[int(s_op)],'probability':float(prob),'magnitude':float(mag_mean),'applied':False,'parameter':None}
        self.current=row
        result=original_apply(self,x,prob,s_op,mag_mean)
        self.trace.append(row)
        return result
    obj._apply_op=types.MethodType(logged_apply,obj)
    x=torch.zeros((len(scores),3,32,32),dtype=torch.float32)
    output=obj(x,torch.tensor(scores,dtype=torch.float64))
    assert output.shape==x.shape and torch.equal(output,x) # Placeholder only, not a pixel-effect claim.
    assert len(obj.trace)==2*len(scores)
    return obj.trace

def main():
    assert sha(HERE/'protocol.json')==json.loads((HERE/'protocol-lock.json').read_text())['protocol_sha256']
    for f,h in json.loads((HERE/'code-lock.json').read_text())['files'].items():assert sha(HERE/f)==h
    source_manifest=json.loads((HERE/'source-manifest.json').read_text())
    for r in source_manifest:
        p=HERE/'source'/r['path'];raw=p.read_bytes()
        assert sha(p)==r['sha256']
        assert hashlib.sha1(('blob '+str(len(raw))).encode()+bytes([0])+raw).hexdigest()==r['git_blob_sha']
    old=ROOT/CFG['prior_audit_dir']
    monitored=[*(old.rglob('*')),ROOT/CFG['input'],ROOT/CFG['manifest'],ROOT/'augmentation-research-20261022/pilot-20261022/protocol.json',ROOT/'augmentation-research-20261022/pilot-20261022/train.py',ROOT/'augmentation-research-20261022/pilot-20261022/common.py',HERE/'protocol.json',HERE/'protocol-lock.json',HERE/'source-manifest.json',HERE/'history.json',HERE/'impact.py',HERE/'code-lock.json',*(HERE/'source').rglob('*')]
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in monitored if p.is_file()}
    with np.load(ROOT/CFG['input'],allow_pickle=False) as arr:
        z=arr['logits'].astype(np.float64);y=arr['labels'].astype(np.int64);ids=arr['source_ids'].astype(str)
    manifest=json.loads((ROOT/CFG['manifest']).read_text())
    dev={r['source_id']:r['class_id'] for r in manifest['rows'] if r['candidate_partition']=='development'}
    assert z.shape==(589,37) and set(ids)==set(dev) and all(dev[s]==int(k) for s,k in zip(ids,y))
    counts=collections.Counter(r['candidate_partition'] for r in manifest['rows'])
    assert counts=={'train':591,'development':589}
    batches={
      'official_CIFAR10_default':summary_batches(50000,256,10),
      'official_CIFAR100_same_advertised_recipe':summary_batches(50000,256,100),
      'official_ImageNet_default':summary_batches(1281167,512,1000,world=4),
      'local_Pets_actual_train_plain_RA':summary_batches(591,16,37,active=False),
      'local_Pets_actual_dev_plain_RA':summary_batches(589,16,37,active=False),
      'hypothetical_supported_CIFAR10_total_batch20':summary_batches(50000,20,10),
      'constructed_supported_CIFAR10_tail_N50004_batch64':summary_batches(50004,64,10),
    }
    # Cross-check batching using actual public BatchSampler and DistributedSampler, no dataset IO.
    from torch.utils.data import BatchSampler,SequentialSampler,TensorDataset,DataLoader
    from torch.utils.data.distributed import DistributedSampler
    sampler_checks={}
    for name,spec in batches.items():
        if spec['world']>1:
            sample=DistributedSampler(range(spec['N']),num_replicas=spec['world'],rank=0,shuffle=False,drop_last=False)
        else:sample=SequentialSampler(range(spec['N']))
        lens=[len(ix) for ix in BatchSampler(sample,spec['per_rank_full_batch'],False)]
        assert len(lens)==spec['loader_batch_count'] and lens[-1]==spec['last_loader_batch']
        assert dict(collections.Counter(n-n//2 for n in lens))==spec['scoring_batch_histogram']
        sampler_checks[name]=True
    # Direct collate of class-index metadata establishes the native integer target shape.
    meta=TensorDataset(torch.arange(len(y)),torch.from_numpy(y))
    _,target=next(iter(DataLoader(meta,batch_size=74,shuffle=False,num_workers=0)))
    assert target.dtype==torch.int64 and target.shape==(74,) and target[37:].shape==(37,)
    node=getmethod('models/cifar/preact_resnet.py','ResNet','forward')
    original=compilemethod(node,{'torch':torch,'np':np,'F':F})
    fixed_node=copy.deepcopy(node)
    matched=[n for n in ast.walk(fixed_node) if isinstance(n,ast.If) and ast.unparse(n.test)=='y.shape[-1] != ori.shape[-1]']
    assert len(matched)==1
    old_test=ast.dump(matched[0].test)
    matched[0].test=ast.parse('y.ndim == 1',mode='eval').body
    patched=compilemethod(fixed_node,{'torch':torch,'np':np,'F':F})
    restored=copy.deepcopy(fixed_node)
    target_if=next(n for n in ast.walk(restored) if isinstance(n,ast.If) and ast.unparse(n.test)=='y.ndim == 1')
    target_if.test=copy.deepcopy(next(n for n in ast.walk(node) if isinstance(n,ast.If) and ast.unparse(n.test)=='y.shape[-1] != ori.shape[-1]').test)
    assert ast.dump(restored)==ast.dump(node)
    adapter_spec=importlib.util.spec_from_file_location('prior_mis_adapter',ROOT/CFG['prior_adapter']);adapter=importlib.util.module_from_spec(adapter_spec);adapter_spec.loader.exec_module(adapter)
    groups=json.loads((ROOT/CFG['prior_group_mapping']).read_text())
    score_rows=[];trace_rows=[];rank_rows=[];max_control=0.;endpoint=CFG['tolerances']['score_atol'];ptol=CFG['tolerances']['parameter_atol']
    for group in groups:
        g=group['group'];ix=np.asarray(group['indices']);tz=torch.from_numpy(z[ix]);ty=torch.from_numpy(y[ix])
        s0=score_via_forward(original,tz,ty).numpy();s1=score_via_forward(patched,tz,ty).numpy();s2=formula(z[ix],y[ix])
        sa=adapter.mis_score(tz,ty).numpy();so=score_via_forward(original,tz,F.one_hot(ty,37).to(torch.float64)).numpy()
        error=max(float(np.max(np.abs(s1-s2))),float(np.max(np.abs(sa-s2))),float(np.max(np.abs(so-s2))))
        assert error<=endpoint;max_control=max(max_control,error)
        if len(ix)!=37:assert np.max(np.abs(s0-s2))<=endpoint
        b=len(ix);ids_g=ids[ix]
        r0=np.empty(b,dtype=int);r1=np.empty(b,dtype=int)
        r0[np.lexsort((ids_g,s0))]=np.arange(b);r1[np.lexsort((ids_g,s1))]=np.arange(b)
        inv=ties=0
        for i in range(b):
            for j in range(i+1,b):
                a=s0[i]-s0[j];bb=s1[i]-s1[j]
                if a==0 or bb==0:ties+=1
                elif a*bb<0:inv+=1
        k=math.ceil(b/4);overlap=len(set(np.flatnonzero(r0>=b-k))&set(np.flatnonzero(r1>=b-k)))
        rank_rows.append({'group':g,'B':b,'collision':b==37,'pairs':b*(b-1)//2,'pair_inversions':inv,'pair_ties':ties,'spearman_id_tiebroken':1-6*float(np.sum((r0-r1)**2))/(b*(b*b-1)),'top_quartile_size':k,'top_quartile_overlap':overlap,'mean_abs_rank_displacement':float(np.mean(np.abs(r0-r1)))})
        traces=[operation_trace(scores,g) for scores in (s0,s1,s2)]
        for j,i in enumerate(ix):
            score_rows.append({'source_id':ids[i],'label':int(y[i]),'group':g,'B':b,'collision':b==37,'original':float(s0[j]),'patched':float(s1[j]),'oracle':float(s2[j]),'prior_adapter':float(sa[j]),'one_hot':float(so[j]),'rank_original':int(r0[j]),'rank_patched':int(r1[j])})
        for row0,row1,row2 in zip(*traces):
            assert all(row0[k]==row1[k]==row2[k] for k in ('sample','depth','operation','probability','applied'))
            assert row0['applied'] and row0['probability']==1
            delta=None if row0['parameter'] is None else row1['parameter']-row0['parameter']
            if delta is not None:assert abs(row1['parameter']-row2['parameter'])<=ptol
            i=ix[row0['sample']]
            trace_rows.append({'source_id':ids[i],'group':g,'B':b,'collision':b==37,'depth':row0['depth'],'operation':row0['operation'],'applied':row0['applied'],'probability':row0['probability'],'magnitude_original':row0['magnitude'],'magnitude_patched':row1['magnitude'],'magnitude_oracle':row2['magnitude'],'parameter_original':row0['parameter'],'parameter_patched':row1['parameter'],'parameter_oracle':row2['parameter'],'argument_changed':delta is not None and abs(delta)>ptol,'parameter_delta':delta})
    shadow=[]
    for fullbatch in (16,256):
        order=np.concatenate([np.asarray(g['indices']) for g in groups]);errors=[];hits=0;calls=[]
        for start in range(0,len(order),fullbatch):
            indices=order[start:start+fullbatch];ix=indices[len(indices)//2:]
            a=score_via_forward(original,torch.from_numpy(z[ix]),torch.from_numpy(y[ix])).numpy()
            b=score_via_forward(patched,torch.from_numpy(z[ix]),torch.from_numpy(y[ix])).numpy()
            c=formula(z[ix],y[ix]);errors.extend(abs(a-c));hits+=int(len(ix)==37);calls.append(len(ix))
            assert np.max(np.abs(b-c))<=endpoint and np.max(np.abs(a-c))<=endpoint
        shadow.append({'full_batch':fullbatch,'N_cache':len(order),'score_calls':calls,'score_records':len(errors),'trigger_batches':hits,'original_vs_oracle':stats(errors),'interpretation':'Hypothetical native SRA second-half scheduling on existing dev cache, not actual training.'})
    coll=[r for r in score_rows if r['collision']];ct=[r for r in trace_rows if r['collision']];rr=[r for r in rank_rows if r['collision']]
    per_op={}
    for op,_,_ in SETTINGS:
        rows=[r for r in ct if r['operation']==op]
        ds=[r['parameter_delta'] for r in rows if r['parameter_delta'] is not None]
        per_op[op]={'selected':len(rows),'applied':sum(r['applied'] for r in rows),'argument_changes':sum(r['argument_changed'] for r in rows),'parameter_delta':stats(ds),'magnitude_sensitive':op in ARG_FUNCTIONS}
    affected={r['source_id'] for r in ct if r['argument_changed']}
    outcome={'status':'completed','at_utc':datetime.now(timezone.utc).isoformat(),'input_contract':{'original_forward_accepts_int64_B_vector':True,'original_forward_hard_vs_onehot_mismatch_confirmed':True,'encoder_is_fixed_logit_stub':True,'images_are_zero_placeholders':True,'only_repair_is_shape_condition':True,'prior_adapter_matches_same_method_repair':True,'original_dtype_device_contract':'existing CPU FP64 logits, int64 hard targets, synthetic float32 Bx3x32x32 image shape','no_original_network_layers_or_pixel_ops_executed':True},'loader_schedules':batches,'native_sampler_checks':sampler_checks,'shadow_cache_schedules':shadow,'score_control_max_abs':max_control,'collision_score_delta':stats([r['patched']-r['original'] for r in coll]),'ranking':{'pair_inversions':sum(r['pair_inversions'] for r in rr),'total_pairs':sum(r['pairs'] for r in rr),'pair_ties':sum(r['pair_ties'] for r in rr),'weighted_inversion_fraction':sum(r['pair_inversions'] for r in rr)/sum(r['pairs'] for r in rr),'mean_group_spearman':sum(r['spearman_id_tiebroken'] for r in rr)/len(rr),'top_quartile_overlap':sum(r['top_quartile_overlap'] for r in rr),'top_quartile_slots':sum(r['top_quartile_size'] for r in rr),'is_used_by_SRA_for_selection':False},'augmentation_decisions':{'all_slots':len(trace_rows),'collision_slots':len(ct),'selected_operator_differences':0,'application_gate_differences':0,'collision_argument_changed_slots':sum(r['argument_changed'] for r in ct),'collision_samples_with_argument_change':len(affected),'collision_samples_total':len(coll),'per_operator':per_op,'trace_uses_actual_script_probabilities':[1.,1.],'no_pixel_output_or_accuracy_effect_claim':True},'ranking_per_group':rank_rows,'training_decision':'archive_as_verified_implementation_case_no_downstream_training: inspected actual/default configurations have zero trigger exposure; forced probes show only conditional impact','actual_config_with_trigger_found':False,'resources':{'wall_seconds_including_imports_before_write':time.perf_counter()-START,'peak_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'torch':torch.__version__,'numpy':np.__version__,'training_runs':0,'neural_network_forwards':0,'image_decodes':0,'actual_pixel_ops':0,'gpu_calls':0},'input_hashes':input_hashes,'old_audit_and_source_hashes_unchanged':all(sha(ROOT/p)==h for p,h in input_hashes.items())}
    assert outcome['old_audit_and_source_hashes_unchanged']
    assert not any(v['triggering_batches'] for k,v in batches.items() if k.startswith(('official_','local_')))
    csvout('scores.csv',score_rows);csvout('decisions.csv',trace_rows);dump('result.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k not in ('input_hashes','ranking_per_group','loader_schedules','augmentation_decisions')},ensure_ascii=False,indent=2))
    print(json.dumps(outcome['augmentation_decisions'],ensure_ascii=False,indent=2))
    signal.alarm(0)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if not (HERE/'failure.json').exists():dump('failure.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'type':type(e).__name__,'message':str(e),'wall_seconds':time.perf_counter()-START})
        raise
