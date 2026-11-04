import json
import os
import stat
import time
from pathlib import Path
import numpy as np
import torch
from .protocol import ASSETS,WORK,SPEC,THETAS,file_sha,save_json
from .data import read_rgb
from .encoder import encode,transform,color_target,color_normalize
from .methods import fit_bow,fit_projections

def disk_bytes():
    return sum(p.lstat().st_blocks*512 for p in ASSETS.rglob('*') if stat.S_ISREG(p.lstat().st_mode))

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    return {'path':str(path.relative_to(WORK)),'bytes':path.stat().st_size,'sha256':file_sha(path)}

def save_array(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    return {'path':str(path.relative_to(WORK)),'bytes':path.stat().st_size,'sha256':file_sha(path)}

def save_query_cache(path,A,argq,projected,color_A,color_argq):
    arrays={'A':A,'argmin_q':argq,'color_A':color_A,'color_argmin_q':color_argq}
    for name,modes in projected.items():
        for mode,r in modes.items():
            key=name.replace('/','_')+'_'+mode
            arrays[key+'_distance']=r['distances'];arrays[key+'_winner']=r['indices'].astype(np.int32)
    return save_npz(path,**arrays)

class ComputeWindow:
    """Persistent first-GPU wall clock; never reset after time spent on CPU/waiting."""
    def __init__(self,preflight=False):
        self.path=WORK/'reports/gpu-window.json';self.preflight=preflight;self.begun=False
    def begin(self):
        if self.path.exists():
            d=json.loads(self.path.read_text())
            if self.preflight:raise RuntimeError('Preflight MPS window already started; no automatic fresh window')
            self.start=d['first_gpu_epoch_seconds']
        else:
            self.start=time.time()
            save_json(self.path,{'first_gpu_epoch_seconds':self.start,'first_gpu_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(self.start)),
              'preflight_deadline_epoch_seconds':self.start+300,'formal_total_deadline_epoch_seconds':self.start+7200,
              'includes_cpu_wait_and_review_wait':True,'preflight_limit_seconds':300,'total_limit_seconds':7200})
        self.begun=True;self.check()
    def elapsed(self):return time.time()-self.start
    def check(self,reserve=0):
        if not self.begun:return
        limit=300 if self.preflight else 7200
        if self.elapsed()+reserve>=limit:raise TimeoutError(f'Compute wall window exhausted ({limit}s)')
    def check_disk(self):
        if disk_bytes()>SPEC['budgets']['disk_bytes']:raise RuntimeError('8 GiB disk budget exceeded; stop, do not prune baselines')

def sync(device):
    if str(device)=='mps':torch.mps.synchronize()

def prepare_seed(model,manifest,seed,device,window,output_dir):
    """Only reference images. Timings separate preparation, encoding and baseline fits."""
    refs=manifest['references'][str(seed)];ids=[c['cluster_id'] for c in refs];timings={}
    t=time.perf_counter();rgbs=[read_rgb(ASSETS/'data'/c['canonical']['path']) for c in refs];timings['ref_decode']=time.perf_counter()-t
    features=[];prep=0.;forward=0.
    for rgb in rgbs:
        window.check();t=time.perf_counter();views=[transform(rgb,x) for x in THETAS];prep+=time.perf_counter()-t
        t=time.perf_counter();features.append(encode(model,views,device,batch_size=9,deadline_check=window.check));sync(device);forward+=time.perf_counter()-t
    bank=torch.stack(features,dim=0);timings['ref_view_prepare']=prep;timings['ref_encode_576']=forward
    window.check();t=time.perf_counter();target=color_target(rgbs);timings['color_target_fit']=time.perf_counter()-t
    t=time.perf_counter();normalized=[color_normalize(x,target) for x in rgbs];timings['color_ref_prepare']=time.perf_counter()-t
    t=time.perf_counter();cb=encode(model,[x[0] for x in normalized],device,batch_size=8,deadline_check=window.check);sync(device);timings['color_ref_encode_64']=time.perf_counter()-t
    window.check();t=time.perf_counter();bases,evidence=fit_projections(bank.numpy());timings['projection_fit']=time.perf_counter()-t
    window.check();t=time.perf_counter();centers,rh=fit_bow(bank.numpy());timings['bow_fit_and_reference_hist']=time.perf_counter()-t
    window.check();t=time.perf_counter()
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    artifacts=[save_array(output_dir/'bank.npy',bank.numpy()),save_array(output_dir/'color-bank.npy',cb.numpy())]
    state={'centers':centers,'ref_hist':rh,'target_median':target[0],'target_iqr':target[1],'delta_mu':evidence['mu']}
    for name in bases:
        state[name+'_basis']=bases[name];state[name+'_basis_float64']=evidence[name]['basis_float64'];state[name+'_eigenvalues']=evidence[name]['eigenvalues'];state[name+'_matrix']=evidence[name]['matrix']
    artifacts.append(save_npz(output_dir/'fitted-state.npz',**state))
    timings['setup_persistence']=time.perf_counter()-t
    return {'bank':bank.to(device),'color_bank':cb[:,None,:,:].to(device),'bases':bases,'centers':centers,'ref_hist':rh,'ref_ids':ids,
      'target':target,'timings':timings,'artifacts':artifacts,'color_floor_channels_refs':sum(x[1] for x in normalized),
      'reference_source_paths':[c['canonical']['path'] for c in refs]}

def verify_frozen():
    frozen=json.loads((WORK/'frozen-manifest.json').read_text())
    for f in frozen['code_files']:
        if file_sha(WORK/f['path'])!=f['sha256']:raise RuntimeError('Code changed after freeze: '+f['path'])
    if file_sha(WORK/'split-manifest.json')!=frozen['split_manifest_sha256']:raise RuntimeError('Split manifest changed after freeze')
    if file_sha(WORK/'shuffle-mappings.json')!=frozen['shuffle_mappings_sha256']:raise RuntimeError('Shuffle mapping artifact changed after freeze')
    return frozen
