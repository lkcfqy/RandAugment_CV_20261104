"""One support-fit or query phase; original encoders and ridge primitives reused."""
import os,sys,time,importlib.util,resource
from contracts import *
phase,model_name=sys.argv[1:];assert phase in ['fit','query'] and model_name in MODELS
source=PROJECT/('augmentation-selection-pilot-20261102' if model_name=='dinov2_small' else 'augmentation-selection-deit-20261102')
spec=importlib.util.spec_from_file_location('frozen_common',source/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
proto=read(PREFLIGHT/'PROTOCOL.json');ids=read(PREFLIGHT/'IDENTITIES.json');limits=read(ROOT/'LIMITS.json')
assert sha(PREFLIGHT/'PROTOCOL.json')==limits['protocol_sha256'] and sha(PREFLIGHT/'IDENTITIES.json')==limits['identity_sha256']
assert sha(source/'common.py')==limits['common_sha256'][model_name]
c.SALT=proto['sampling_and_view_salt'];D=MODELS[model_name];out=ROOT/model_name;out.mkdir(exist_ok=True)
guard=Guard();start=time.monotonic();guard.check()
raw=np.load(PROJECT/'research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy',mmap_mode='r',allow_pickle=False)
if phase=='query':
    gate=read(ROOT/'JOINT_QUERY_GATE.json')
    assert gate['status']=='BOTH_ENCODERS_SEALED'
    assert validate_joint_gate(ROOT,limits['protocol_sha256'],limits['identity_sha256'])==gate['fit_seal_sha256']
model=c.load_model();state_before=c.state_hash(model)
expected=read(PREFLIGHT/(model_name+'_BENCHMARK.json'))['initial_state_hash'];assert state_before==expected
entries=ids['support_union'] if phase=='fit' else ids['query'];n=len(entries);views=9 if phase=='fit' else 1
basename='support_features' if phase=='fit' else 'query_features'
tmp=out/(basename+'.inprogress.npy');final=out/(basename+'.npy')
assert not final.exists() and not tmp.exists(),'Never overwrite a completed or partial phase'
features=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(n,views,D))
feature_start=time.monotonic()
for first in range(0,n,16):
    guard.check();chunk=entries[first:first+16];tensors=[]
    for r in chunk:
        image=c.Image.fromarray(raw[r['index']].copy())
        tensors.extend(c.make_views(image,r['rgb_hash'],all_views=phase=='fit'))
    value=c.encode(model,tensors,batch=16).reshape(len(chunk),views,D)
    features[first:first+len(chunk)]=value
    if first%128==0 or first+len(chunk)==n:
        features.flush();save(ROOT/'PROGRESS.json',{'phase':phase,'encoder':model_name,'completed_originals':first+len(chunk),'total_originals':n,'phase_elapsed_seconds':time.monotonic()-start,'remaining_seconds':guard.remaining(),'query_accuracy_computed':False})
feature_seconds=time.monotonic()-feature_start
finish_memmap(features,tmp,final)
state_after=c.state_hash(model);assert state_before==state_after
del model,tensors,value,features
features=np.load(final,mmap_mode='r',allow_pickle=False)
heads=canonical_heads()
if phase=='fit':
    index={r['id']:i for i,r in enumerate(ids['support_union'])}
    W=np.empty((360,D,100),dtype=np.float64);b=np.empty((360,100),dtype=np.float64)
    cv_head=[];cv_id=[];cv_label=[];cv_prediction=[];choices=[];fit_times=[]
    for e in ids['episodes']:
        allix=np.array([index[i] for i in e['ids_16']],dtype=np.int64)
        labels=np.array(e['labels_16']);ranks=np.array(e['rank_16']);folds=np.array(e['fold_16'])
        for B in [8,16]:
            mask=ranks<B;ix=allix[mask];y=labels[mask];fold=folds[mask];totals=[];full_indices=[]
            for policy in POLICIES:
                x=c.policy_features(features[ix],policy);correct=0
                for f in [0,1,2,3,-1]:
                    guard.check();h=next(h['index'] for h in heads if (h['episode'],h['B'],h['policy'],h['fold'])==(e['episode'],B,policy,f))
                    train=np.ones(len(y),bool) if f==-1 else fold!=f
                    t=time.monotonic();W[h],b[h]=c.ridge(x[train],y[train],classes=100);fit_times.append(time.monotonic()-t)
                    if f==-1:full_indices.append(h)
                    else:
                        held=fold==f;pred=c.scores(features[ix[held],0],W[h],b[h]).argmax(1).astype(np.uint8)
                        correct+=int((pred==y[held]).sum());cv_head.extend([h]*int(held.sum()));cv_id.extend(ix[held].tolist());cv_label.extend(y[held].tolist());cv_prediction.extend(pred.tolist())
                totals.append(correct)
            selected=c.choose(totals)
            choices.append({'episode':e['episode'],'B':B,'cv_correct':totals,'cv_total':B*100,'selected_index':selected,'selected_policy':POLICIES[selected],'full_head_indices':full_indices})
            save(ROOT/'PROGRESS.json',{'phase':'fitting','encoder':model_name,'completed_choices':len(choices),'total_choices':24,'remaining_seconds':guard.remaining(),'query_accuracy_computed':False})
    durable_array(out/'W.npy',W);durable_array(out/'b.npy',b);save(out/'HEAD_IDENTITIES.json',heads)
    save_npz(out/'CV_PREDICTIONS.npz',head_index=np.array(cv_head,dtype=np.uint16),support_union_index=np.array(cv_id,dtype=np.uint32),label=np.array(cv_label,dtype=np.uint8),prediction=np.array(cv_prediction,dtype=np.uint8))
    files={f:sha(out/f) for f in ['support_features.npy','W.npy','b.npy','HEAD_IDENTITIES.json','CV_PREDICTIONS.npz']}
    save(out/'FIT_SEAL.json',{'status':'FIT_SEALED','protocol_sha256':limits['protocol_sha256'],'identity_sha256':limits['identity_sha256'],'head_count':360,'choices':choices,'files':files,'model_state_before':state_before,'model_state_after':state_after,'utc_epoch':time.time(),'query_accuracy_computed':False})
else:
    W=np.load(out/'W.npy',mmap_mode='r',allow_pickle=False);b=np.load(out/'b.npy',mmap_mode='r',allow_pickle=False)
    tmp=out/'QUERY_PREDICTIONS.inprogress.npy';predictions=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(360,n))
    for h in range(360):
        guard.check()
        for first in range(0,n,256):predictions[h,first:first+256]=c.scores(features[first:first+256,0],W[h],b[h]).argmax(1).astype(np.uint8)
    finish_memmap(predictions,tmp,out/'QUERY_PREDICTIONS.npy')
    save(out/'QUERY_SEAL.json',{'status':'ALL_QUERY_PREDICTIONS_SAVED','protocol_sha256':limits['protocol_sha256'],'identity_sha256':limits['identity_sha256'],'joint_gate_sha256':sha(ROOT/'JOINT_QUERY_GATE.json'),'files':{f:sha(out/f) for f in ['query_features.npy','QUERY_PREDICTIONS.npy']},'model_state_before':state_before,'model_state_after':state_after,'query_images':n,'predictions':360*n,'utc_epoch':time.time(),'accuracy_computed':False})
save(out/(phase.upper()+'_RECEIPT.json'),{'status':'COMPLETE','phase':phase,'encoder':model_name,'wall_seconds':time.monotonic()-start,'feature_seconds':feature_seconds,'image_forwards':n*views,'head_fits':360 if phase=='fit' else 0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'initial_state':state_before,'final_state':state_after,'query_accuracy_computed':False})
print(model_name,phase,'COMPLETE',flush=True)
