"""Complete future runner. No executable CLI; current handoff does not authorize use."""
import json
import os
import threading
import time
from pathlib import Path
import numpy as np
import torch
from stheta.protocol import ASSETS,WORK,SEEDS,save_json,file_sha,query_manifest_records
from stheta.data import read_rgb
from stheta.encoder import load_encoder,encode,color_normalize
from stheta.methods import distance_cache,all_readouts,upper_mean
from stheta.runtime import ComputeWindow,prepare_seed,save_query_cache,save_array,verify_frozen,sync

def run_formal(*,authorized=False,device='mps',group_chunk=32):
    if authorized is not True:
        raise PermissionError('Formal scoring awaits the parent’s explicit next-message confirmation and static review')
    if device!='mps':raise RuntimeError('Only the measured MPS backend is permitted by this resource preflight')
    frozen=verify_frozen();manifest=json.loads((WORK/'split-manifest.json').read_text())
    assert manifest['status']=='ready'
    window=ComputeWindow(preflight=False)
    # A prior preflight start is required and its absolute wall clock must still fit.
    if not window.path.exists():raise RuntimeError('No completed resource preflight')
    preflight=json.loads((WORK/'reports/throughput-preflight.json').read_text())
    if not preflight.get('full_protocol_fits_120min',False):raise RuntimeError('Resource gate not passed; do not reduce any method or data')
    if preflight['frozen_manifest_sha256']!=file_sha(WORK/'frozen-manifest.json'):raise RuntimeError('Resource preflight predates the current code freeze')
    # Reject stale output before starting a device; resume needs a separately reviewed path.
    output=WORK/'formal'
    if output.exists():raise RuntimeError('Formal output already exists; do not overwrite or reuse unverified partial caches')
    window.begin()
    if window.elapsed()+1.2*preflight['remaining_full_run_plain_seconds']>7200:raise RuntimeError('Review/waiting consumed the remaining forecast budget; no formal run started')
    model=load_encoder();window.check();model.to(device)
    def wall_timeout():
        save_json(WORK/'reports/formal-budget-stop.json',{'status':'stopped_at_absolute_wall_deadline','elapsed':window.elapsed(),'scientific_result_complete':False})
        os._exit(124)
    watchdog=threading.Timer(max(0,7195-window.elapsed()),wall_timeout);watchdog.daemon=True;watchdog.start()
    try:
        return _execute(model,manifest,window,output,device,group_chunk)
    finally:
        model.to('cpu');sync(device);torch.mps.empty_cache();sync(device);watchdog.cancel()

def _execute(model,manifest,window,output,device,group_chunk):
    query_records=query_manifest_records(manifest)
    assert len(query_records)==1026
    output.mkdir();features_dir=output/'identity-query-features';features_dir.mkdir()
    save_json(output/'query-manifest.json',query_records)
    scores_by_seed={};ledger=[];identity_rates={};query_features={}
    for seed in SEEDS:
        state=prepare_seed(model,manifest,seed,device,window,output/f'seed{seed}'/'reference-state')
        ledger.extend(state['artifacts']);columns={};floors=0;choices={}
        for index,record in enumerate(query_records):
            window.check();rgb=read_rgb(ASSETS/'data'/record['path']);fp=features_dir/f'{index:04d}.npy'
            if fp.exists():
                if index not in query_features or file_sha(fp)!=query_features[index]['sha256']:raise RuntimeError('Identity query feature cache mismatch')
                query=torch.from_numpy(np.load(fp,allow_pickle=False)).to(device)
            else:
                query=encode(model,[rgb],device,batch_size=1,deadline_check=window.check)[0].to(device)
                feature=save_array(fp,query.cpu().numpy());feature['query_id']=record['query_id'];query_features[index]=feature;ledger.append(feature)
            A,argq,projected=distance_cache(query,state['bank'],state['bases'],group_chunk=group_chunk,deadline_check=window.check)
            diag={}
            scores=all_readouts(query,state['bank'],A,argq,projected,state['bases'],state['ref_ids'],state['centers'],state['ref_hist'],diagnostics=diag)
            for name,choice in diag.items():choices.setdefault(name,[]).append(choice)
            normalized,n=color_normalize(rgb,state['target']);floors+=n
            cq=encode(model,[normalized],device,batch_size=1,deadline_check=window.check)[0].to(device)
            cA,cix,_=distance_cache(cq,state['color_bank'],group_chunk=group_chunk,deadline_check=window.check)
            scores['robust_color']=float(upper_mean(cA[:,:,0].min(1)))
            assert len(scores)==59
            for name,value in scores.items():columns.setdefault(name,[]).append(value)
            cached=save_query_cache(output/f'seed{seed}'/'distance-cache'/f'{index:04d}.npz',A,argq,projected,cA,cix)
            cached['query_id']=record['query_id'];cached['seed']=seed;ledger.append(cached)
            if (index+1)%16==0:window.check_disk()
            if (index+1)%64==0:
                save_json(output/'cache-ledger.partial.json',ledger)
                save_json(output/'progress.json',{'seed':seed,'queries_completed':index+1,'wall_from_first_gpu':window.elapsed(),'results_final':False})
        scores_by_seed[str(seed)]=columns
        save_json(output/f'seed{seed}'/'scores.json',columns)
        save_json(output/f'seed{seed}'/'color-floor.json',{'references':state['color_floor_channels_refs'],'queries':floors})
        rates={}
        for name,values in choices.items():
            rates[name]={}
            for domain in ('same','illumination'):
                for label in (0,1):
                    ix=[64+i for i,r in enumerate(manifest['tests']) if r['domain']==domain and r['label']==label]
                    rates[name][f'{domain}/{"N" if label==0 else "A"}']={'identity_count':int(sum(values[i]==0 for i in ix)),'n':len(ix),'identity_rate':float(np.mean([values[i]==0 for i in ix]))}
        identity_rates[str(seed)]=rates
        del state
    # This import and all real score metrics are unreachable in preflight mode.
    from stheta.metrics import summarize
    result=summarize(scores_by_seed,query_records,manifest)
    result['identity_selection_by_domain_and_NA']=identity_rates
    save_json(output/'results.json',result);save_json(output/'cache-ledger.json',ledger)
    save_json(output/'completion.json',{'wall_from_first_gpu':window.elapsed(),'frozen_manifest_sha256':file_sha(WORK/'frozen-manifest.json'),'all_methods':59,'all_queries_per_seed':1026,'seeds':SEEDS})
    return result

if __name__=='__main__':
    raise SystemExit('Formal scoring is disabled in this handoff. Await explicit parent confirmation and static review.')
