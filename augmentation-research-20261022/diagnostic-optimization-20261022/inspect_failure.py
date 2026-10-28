"""Post-failure CPU-only description. Never replace the frozen complete analysis."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import sys
ROOT=Path(sys.argv[1])
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def read(p):return json.loads(p.read_text())
def state(p,k):return {n:v.numpy().astype(np.float64) for n,v in torch.load(p,map_location='cpu',weights_only=True)[k].items()}
def main():
    cfg=read(ROOT/'protocol.json');lock=read(ROOT/'protocol-lock.json');failure=read(ROOT/'diagnostic-failure.json')
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in lock['code_sha256'].items())
    initial={l:state(ROOT/'results'/l/'initial-trainable.pt','trainable_state') for l in ['front','back']}
    cases=[];head={};front=[]
    for folder in sorted((ROOT/'results').glob('*/batch*')):
        if not (folder/'post-step-trainable.pt').exists():continue
        layout=folder.parent.name;g=state(folder/'gradients.pt','gradients');after=state(folder/'post-step-trainable.pt','trainable_state');a0=initial[layout]
        records=[];merged=[]
        for n,x in a0.items():
            lr=cfg['optimizer']['head_actual_lr'] if n.startswith('heads.') else cfg['optimizer']['adapter_actual_lr']
            expected=x*(1-lr*.01)-lr*g[n]/(np.abs(g[n])+1e-8)
            tol=cfg['precision']['manual_AdamW_parameter_oracle'];passed=bool(np.allclose(expected,after[n],atol=tol['atol'],rtol=tol['rtol']))
            records.append({'name':n,'finite':bool(np.isfinite(g[n]).all() and np.isfinite(after[n]).all()),'gradient_nonzero':int(np.count_nonzero(g[n])),
                'gradient_norm':norm(g[n]),'actual_parameter_delta_norm':norm(after[n]-x),'original_AdamW_oracle_passed':passed,
                'oracle_max_abs_error':float(np.max(np.abs(expected-after[n])))})
        for i in cfg['blocks'][layout]:
            prefix=f'encoder.layers.encoder_layer_{i}.self_attention.parametrizations.in_proj_weight.0'
            for q in ['q','v']:merged.append(norm(after[prefix+'.b'+q]@after[prefix+'.a'+q]))
        cases.append({'case':layout+'-'+folder.name,'all_original_AdamW_oracles_passed':all(r['original_AdamW_oracle_passed'] for r in records),
            'all_A_data_gradients_exact_zero':all(r['gradient_nonzero']==0 for r in records if r['name'].endswith(('.aq','.av'))),
            'all_B_data_gradients_nonzero':all(r['gradient_nonzero']>0 for r in records if r['name'].endswith(('.bq','.bv'))),
            'all_finite':all(r['finite'] for r in records),'merged_QV_joint_Frobenius':float(np.sqrt(np.sum(np.square(merged)))),
            'parameters':records})
        if folder.name=='batch0':head[layout]={n:g[n] for n in g if n.startswith('heads.')}
    paired=[]
    for n,a in head['front'].items():
        b=head['back'][n];d=b-a;tol=cfg['precision']['paired_head_gradient_allclose'];allowed=tol['atol']+tol['rtol']*np.abs(a)
        paired.append({'parameter':n,'max_abs_difference':float(np.max(abs(d))),'relative_L2_difference':norm(d)/norm(a),
            'failed_elements_original_tolerance':int(np.sum(abs(d)>allowed)),'total_elements':int(a.size),
            'max_error_over_original_tolerance':float(np.max(abs(d)/allowed))})
    base={}
    for layout in ['front','back']:
        with np.load(ROOT/'results'/layout/'initial-features.npz',allow_pickle=False) as d:
            base[layout]=d['features'].astype(np.float64)
            if layout=='front':repeat=d['repeat_features'].astype(np.float64)
    floor=[float(max(rms(repeat[j]-base['front'][j]),np.finfo(np.float32).eps*rms(base['front'][j]))) for j in range(2)]
    for batch in range(4):
        with np.load(ROOT/'results/front'/f'batch{batch}'/'features.npz',allow_pickle=False) as d:f=d['post_step_features'].astype(np.float64)
        s=[rms(f[j]-base['front'][j]) for j in range(2)]
        front.append({'gradient_batch':batch,'pooled_CLS_change_RMS':rms(f-base['front']),'probe_RMS':s,'probe_floor':floor,'probes_resolved':[s[j]>10*floor[j] for j in range(2)]})
    started=read(ROOT/'results/started.json')
    seconds=(datetime.fromisoformat(failure['at_utc'])-datetime.fromisoformat(started['at_utc'])).total_seconds()
    result={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'partial_diagnostic_preserved_original_failure_no_continuation',
        'executed_isolated_optimizer_steps':len(cases),'completed_case_receipts':len(list((ROOT/'results').glob('*/batch*/receipt.json'))),
        'complete8_case_primary_analysis_available':False,'new_GPU_forward_or_optimizer_calls':0,
        'frozen_protocol_or_tolerance_modified':False,'post_failure_analysis_not_preregistered_new_scientific_test':True,
        'cases':cases,'paired_head_gradient_comparison_batch0':paired,
        'initial_eval_probe_front_back_max_abs_difference':float(np.max(np.abs(base['front']-base['back']))),
        'front_only_feature_changes':front,'front_back_CLS_change_ratio':None,
        'training_batch_CE_before_step':{l:read(ROOT/'results'/l/'batch0/pre-step-loss.json')['CE_repr'] for l in ['front','back']},
        'started_to_failure_UTC_seconds_not_perf_counter_total':seconds,
        'cause':'Not localized. Tiny training-mode FP32/head-gradient discrepancy is observed; backend/autograd-dependent numerics is a hypothesis, not a demonstrated root cause.',
        'decision':'Stop. Do not increase LR, relax tolerance, add batches/seeds or run a full training experiment.'}
    with (ROOT/'partial-analysis.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases']},ensure_ascii=False))
if __name__=='__main__':main()
