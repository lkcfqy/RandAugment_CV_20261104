"""Metadata-only check of ForAug's target-scale scalar. No images or models."""
from array import array
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).parent
SEED = 20261021
K = 14
WIDTH = .1


def info(rows, p_c):
    n = len(rows)
    y_s = [[sum(row[c*K+s] for c in range(n)) for s in range(K)] for row in rows]
    c_s = [[sum(row[c*K+s] for row in rows) for s in range(K)] for c in range(n)]
    p_s = [sum(v[s] for v in y_s) for s in range(K)]
    cmi = 0.0
    marginal_mi = 0.0
    for y,row in enumerate(rows):
        for c in range(n):
            yc = 0.0
            for s in range(K):
                p = row[c*K+s]
                yc += p
                if p:
                    cmi += p*math.log2(p*p_s[s]/(y_s[y][s]*c_s[c][s]))
            if yc:
                marginal_mi += yc*math.log2(yc/(p_c[c]/n))
    yt_mi = sum(p*math.log2(p*n/p_s[s]) for v in y_s for s,p in enumerate(v) if p)
    assert abs(sum(p_s)-1)<1e-10 and abs(marginal_mi)<1e-10
    return {'conditional_mi_bits':max(0.0,cmi),'marginal_y_c_mi_bits':max(0.0,marginal_mi),
            'y_t_mi_bits':yt_mi,'y_t_joint':y_s,'t_marginal':p_s}


def precompute(fg, bg, classes, mode):
    n = len(classes)
    nbg = len(bg)
    kernels=[]
    for i,f in enumerate(fg):
        out=array('d',[0.0])*(n*K)
        rf=float(f['ratio'])
        for b,c in bg:
            rb=float(b['ratio'])
            if mode=='range':
                lo,hi=.7*min(rf,rb),1.3*max(rf,rb)
            else:
                mid=(rf+rb)/2
                lo,hi=.7*mid,1.3*mid
            offset=c*K
            if hi==lo:
                out[offset+min(int(lo/WIDTH),K-1)] += 1/nbg
                continue
            den=(hi-lo)*nbg
            for s in range(max(0,int(lo/WIDTH)),min(K-1,int(hi/WIDTH))+1):
                overlap=min(hi,(s+1)*WIDTH)-max(lo,s*WIDTH)
                if overlap>0:
                    out[offset+s] += overlap/den
        assert abs(sum(out)-1)<1e-10
        kernels.append(out)
        if (i+1)%800==0:
            print(json.dumps({'phase':mode+'_kernel','foreground_rows':i+1}),flush=True)
    return kernels


def aggregate(kernels, order, nclasses, per_class):
    factor=1/(nclasses*per_class)
    return [[sum(v)*factor for v in zip(*(kernels[i] for i in order[y*per_class:(y+1)*per_class]))]
            for y in range(nclasses)]


def main():
    for name in ['scale-results.json','scale-range-joint.json']:
        assert not (ROOT/name).exists(),'Preserving existing '+name
    try:
        os.nice(10)
    except OSError:
        pass
    wall0,cpu0=time.perf_counter(),time.process_time()
    prior=json.loads((ROOT/'results.json').read_text())
    classes=prior['primary_classes']
    n=len(classes)
    idx={c:i for i,c in enumerate(classes)}
    split=list(csv.DictReader((ROOT/'primary-split.csv').open(newline='')))
    fg=[r for c in classes for r in split if r['source_class']==c and r['fold']=='foreground' and r['used_in_primary']=='1']
    bg=[(r,idx[r['source_class']]) for r in split if r['fold']=='background' and r['used_in_primary']=='1']
    assert not ({r['source_id'] for r in fg}&{r['source_id'] for r,c in bg})
    per_class=len(fg)//n
    assert len(fg)==2400 and len(bg)==2223 and per_class==24
    counts=Counter(c for _,c in bg)
    p_c=[counts[c]/len(bg) for c in range(n)]
    order=list(range(len(fg)))
    kernels=precompute(fg,bg,classes,'range')
    joint=aggregate(kernels,order,n,per_class)
    primary=info(joint,p_c)
    # Independent shadow background produces same P(Y,T), exact per-image BG marginal.
    shadow=[[primary['y_t_joint'][y][s]*p_c[c] for c in range(n) for s in range(K)] for y in range(n)]
    control=info(shadow,p_c)
    margin_error=max(abs(a-b) for ra,rb in zip(primary['y_t_joint'],control['y_t_joint']) for a,b in zip(ra,rb))
    assert margin_error<1e-12 and control['conditional_mi_bits']<1e-12
    print(json.dumps({'phase':'range_primary','conditional_mi_bits':primary['conditional_mi_bits']}),flush=True)
    rng=random.Random(SEED+4200)
    null=[]
    for trial in range(99):
        rng.shuffle(order)
        null.append(info(aggregate(kernels,order,n,per_class),p_c)['conditional_mi_bits'])
        if (trial+1)%33==0:
            print(json.dumps({'phase':'range_permutation','complete':trial+1}),flush=True)
    del kernels
    mean_kernels=precompute(fg,bg,classes,'mean')
    mean_point=info(aggregate(mean_kernels,list(range(len(fg))),n,per_class),p_c)
    def compact(data):
        return {k:v for k,v in data.items() if k not in ('y_t_joint','t_marginal')}
    result={
        'status':'completed_requested_scale_scalar_mechanism_only',
        'primary_variant':'range (constructor default in pinned source)',
        'primary':compact(primary),'shadow_scale_control':compact(control),
        'mean_variant_point':compact(mean_point),
        'permutation_reference':{'count':99,'values_bits':null,'max_bits':max(null),'mean_bits':sum(null)/99,
                                 'rank_fraction':(1+sum(v>=primary['conditional_mi_bits'] for v in null))/100},
        'local_precommitted_signal_passed':primary['conditional_mi_bits']>=.005 and primary['conditional_mi_bits']>max(null),
        'target_scale_bin_width':WIDTH,'target_scale_bin_count':K,
        'foreground_count':len(fg),'background_count':len(bg),'classes':classes,
        'source_overlap':0,'y_t_margin_max_difference':margin_error,
        'background_source_class_marginal':p_c,
        'interpretation':'Metadata/source-label conditional dependence of requested scale before alpha adjustment, rasterization and crop. Not visual shortcut or harmful model behavior evidence.',
        'foraug_source':'https://github.com/tobna/ForAug/blob/fca8765fe8e5c3f99ee8d5e426b87d06b1c817cb/fornet.py#L298',
        'causal_scope':'Intervention on the scalar generation program only. No claim of a real-world causal mechanism.',
        'execution':{'wall_seconds':time.perf_counter()-wall0,'cpu_seconds':time.process_time()-cpu0,
                     'python':sys.version,'cpu_workers':1,'ARIS_used':False,'GPU_or_model_used':False,
                     'image_data_read':False,'new_data_downloads':0,
                     'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'protocol_sha256':hashlib.sha256((ROOT/'scale-followup-protocol.md').read_bytes()).hexdigest(),
                     'input_split_sha256':hashlib.sha256((ROOT/'primary-split.csv').read_bytes()).hexdigest()}}
    for name,obj in [('scale-results.json',result),('scale-range-joint.json',{'classes':classes,'p_c':p_c,'bins':{'width':WIDTH,'count':K},'joint_rows':joint})]:
        with (ROOT/name).open('x') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2)
            f.write('\n')
    print(json.dumps({'status':result['status'],'primary':result['primary'],'control':result['shadow_scale_control'],
                      'mean_point':result['mean_variant_point'],'null_max':max(null),'rank':result['permutation_reference']['rank_fraction'],
                      'signal_passed':result['local_precommitted_signal_passed'],'wall_seconds':result['execution']['wall_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
