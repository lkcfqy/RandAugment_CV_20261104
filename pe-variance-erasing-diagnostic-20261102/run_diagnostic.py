"""Frozen CPU erasure/PE diagnosis. No optimizer, network, or inference selection."""
from common import *
import gc,inspect,platform,signal,traceback
import transformers,safetensors,PIL

EXPECTED_PROTOCOL='41cdf66ccac51c83b9c2645606fd68fcff2d7a2f5ce4553e0967332b1e2e6dfa'

def environment():
    modules={'torch':torch,'torchvision':torchvision,'timm':timm,'transformers':transformers,'safetensors':safetensors,'numpy':np,'Pillow':PIL}
    sources=[Path(inspect.getfile(RandomErasing)),Path(timm.__file__).parent/'models/vision_transformer.py',Path(transformers.__file__).parent/'models/dinov2/modeling_dinov2.py',Path(torchvision.__file__).parent/'transforms/transforms.py']
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'device':'cpu','torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
      'modules':{k:{'version':v.__version__,'path':v.__file__} for k,v in modules.items()},'sources':{str(s):sha(s) for s in sources},'offline':True}

class Capture:
    def __init__(self,norm,patch_module,pos_module,position):
        self.alpha=1.;self.position=position;self.cache={}
        def patch(m,args,out):self.cache['patch']=out.detach().clone()
        def position_hook(m,args,out):return out if self.alpha==1. else out+(self.alpha-1.)*self.position
        def pre(m,args):self.cache['norm_input']=args[0].detach().clone()
        def post(m,args,out):self.cache['norm_output']=out.detach().clone()
        self.handles=[patch_module.register_forward_hook(patch),pos_module.register_forward_hook(position_hook),norm.register_forward_pre_hook(pre),norm.register_forward_hook(post)]
    def close(self):
        for h in self.handles:h.remove()

def run():
    proto=read('PROTOCOL.json');assert sha(P/'PROTOCOL.json')==EXPECTED_PROTOCOL
    assert read('TEST_RESULTS.json')['passed']
    assert not (P/'RUN_FROZEN.json').exists(),'Do not silently restart an existing run.'
    check_budget(proto)
    frozen={'UTC':utc(),'protocol_sha256':EXPECTED_PROTOCOL,'code':{q.name:sha(q) for q in P.glob('*.py')},'environment':environment(),'status':'FROZEN_BEFORE_MODEL_CONSTRUCTION','scientific_revision':0}
    dump('RUN_FROZEN.json',frozen)
    start=time.monotonic();counts={'core':0,'identity':0,'halfstep':0};identities=[];receipts={};forward_seconds=0.;save_seconds=0.
    def progress(status,**kwargs):
        dump('PROGRESS.json',{'UTC':utc(),'status':status,'full_forwards':counts,'identities_completed':len(identities),'elapsed_seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**kwargs})
    def timeout(signum,frame):raise TimeoutError('Frozen whole-task wall deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(proto['deadline_epoch']-time.time())))
    try:
        with torch.inference_mode():
            for name in proto['models']:
                check_budget(proto);model,norm,patchmod,posmod,pos,patchsize,receipt=load_model(name,proto)
                if name=='deit_tiny':assert isinstance(patchmod.norm,torch.nn.Identity)
                metadata_hash=npz(f'raw/{name}/MODEL_COMPONENTS.npz',position=pos.numpy(),norm_weight=norm.weight.numpy(),norm_bias=norm.bias.numpy(),norm_eps=np.array(norm.eps),patch_size=np.array(patchsize))
                receipt.update(component_npz_sha256=metadata_hash,architecture=str(type(model)),first_norm=str(norm),patch_module=str(patchmod),position_shape=list(pos.shape))
                first=load_image(proto['images'][0]);t=time.monotonic();original=outputs(model,name,first[None]);forward_seconds+=time.monotonic()-t;counts['identity']+=1
                capture=Capture(norm,patchmod,posmod,pos)
                image_times=[];max_contract=0.;max_norm=0.;half_checks=[]
                try:
                    for ix,row in enumerate(proto['images']):
                        check_budget(proto);image_start=time.monotonic();xs=variants(load_image(row),row)
                        rec={'model':name,'image_id':row['id'],'image_sha256':row['sha256'],'rectangle_tlhw':row['rectangle_tlhw'],'modes':{}}
                        patches=[];features=[];logits=[];pervar=[];perS=[];half_features=[]
                        for mode in proto['modes']:
                            x=xs[mode];fs=[];ls=[];saved_patch=None
                            for alpha in proto['alphas']:
                                check_budget(proto);capture.alpha=alpha;t=time.monotonic();f,l=outputs(model,name,x[None]);dt=time.monotonic()-t;forward_seconds+=dt;counts['core']+=1
                                assert torch.isfinite(f).all() and (l is None or torch.isfinite(l).all())
                                fs.append(f[0].numpy().copy())
                                if l is not None:ls.append(l[0].numpy().copy())
                                xp=capture.cache['patch'][0]
                                expected=xp+alpha*pos[0,1:]
                                err=float((capture.cache['norm_input'][0,1:]-expected).abs().max());max_contract=max(max_contract,err);assert err<2e-5,(name,alpha,err)
                                actual_ln=capture.cache['norm_output'];manual_ln=F.layer_norm(capture.cache['norm_input'],norm.normalized_shape,norm.weight,norm.bias,norm.eps)
                                ne=float((actual_ln-manual_ln).abs().max());max_norm=max(max_norm,ne);assert ne<1e-6
                                if alpha==1.:
                                    saved_patch=xp.clone()
                                    if ix==0 and mode=='none':
                                        assert torch.equal(f,original[0])
                                        assert l is None or torch.equal(l,original[1])
                                        receipt['alpha1_hook_identity_bitwise']=True
                            stats,pv,ps=ln_statistics(saved_patch,pos[0,1:],norm,row,patchsize)
                            rec['modes'][mode]={'input':input_stats(x,row),'tokens':stats,'patch_tensor_sha256':th(saved_patch)}
                            patches.append(saved_patch.numpy().copy());features.append(np.stack(fs));pervar.append(pv);perS.append(ps)
                            if ls:logits.append(np.stack(ls))
                            if ix==0:
                                hf=[]
                                for alpha in (.995,1.005):
                                    capture.alpha=alpha;t=time.monotonic();f,l=outputs(model,name,x[None]);forward_seconds+=time.monotonic()-t;counts['halfstep']+=1
                                    assert torch.isfinite(f).all();hf.append(f[0].numpy().copy())
                                half_features.append(np.stack(hf));d1=(fs[2].astype('float64')-fs[0])/.02;d2=(hf[1].astype('float64')-hf[0])/.01
                                half_checks.append({'mode':mode,'derivative_vector_relative_difference':float(np.linalg.norm(d2-d1)/(np.linalg.norm(d2)+1e-30))})
                        arrays={'patches':np.stack(patches),'features':np.stack(features),'token_feature_variance':np.stack(pervar),'token_LN_PE_sensitivity_RMS':np.stack(perS)}
                        if logits:arrays['logits']=np.stack(logits)
                        if half_features:arrays['halfstep_features']=np.stack(half_features)
                        t=time.monotonic();rel=f'raw/{name}/{ix:02d}_{row["id"]}.npz';rec['npz_path']=rel;rec['npz_sha256']=npz(rel,**arrays)
                        rec['seconds_including_save']=time.monotonic()-image_start
                        dump(rel.replace('.npz','.json'),rec);save_seconds+=time.monotonic()-t
                        identities.extend((name,row['id'],m) for m in proto['modes']);image_times.append(time.monotonic()-image_start)
                        progress('RUNNING',model=name,image_index_completed=ix+1)
                        print(json.dumps({'model':name,'images_completed':ix+1,'identities':len(identities),'forwards':sum(counts.values()),'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
                        # Conservative first-image throughput gate: current model remaining + next model 4x current cost.
                        if ix==0:
                            remaining=(31+(128 if name=='deit_tiny' else 0))*image_times[0]*1.5+120
                            receipt['first_image_remaining_conservative_seconds']=remaining
                            if time.time()+remaining>=proto['deadline_epoch']:raise RuntimeError('Complete-run throughput estimate exceeds remaining frozen deadline')
                    receipt.update(state_sha256_after=statehash(model),all_gradients_none=all(p.grad is None for p in model.parameters()),max_embedding_contract_abs_error=max_contract,max_first_norm_abs_error=max_norm,halfstep_checks=half_checks,image_seconds=image_times)
                    assert receipt['state_sha256_after']==receipt['state_sha256'] and receipt['all_gradients_none']
                    receipts[name]=receipt;dump(f'raw/{name}/MODEL_RECEIPT.json',receipt)
                finally:capture.close()
                del model,norm,patchmod,posmod,capture,original;gc.collect()
        expected={(n,r['id'],m) for n in proto['models'] for r in proto['images'] for m in proto['modes']}
        assert set(identities)==expected and len(identities)==len(expected)==256
        assert counts=={'core':768,'identity':2,'halfstep':16}
        assert all(sha(v['path'])==v['sha256'] for v in proto['models'].values())
        total=check_budget(proto);progress('COMPLETE')
        dump('RUN_RECEIPT.json',{'UTC':utc(),'status':'COMPLETE','protocol_sha256':EXPECTED_PROTOCOL,'identities':identities,'counts':counts,'model_receipts':receipts,'forward_seconds':forward_seconds,'atomic_save_seconds':save_seconds,'run_seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'artifact_bytes_at_run_end':total,'network_download_install_upload_GPU_MPS_Colab_CU':0,'models_frozen_unchanged':True})
    except BaseException as e:
        progress('INCOMPLETE',error=repr(e));dump('RUN_FAILURE.json',{'UTC':utc(),'error':repr(e),'traceback':traceback.format_exc(),'counts':counts,'completed_identities':identities,'no_subset_analysis':True});raise
    finally:signal.alarm(0)

if __name__=='__main__':run()
