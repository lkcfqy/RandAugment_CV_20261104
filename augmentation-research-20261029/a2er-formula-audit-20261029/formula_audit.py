#!/usr/bin/env python3
"""Deterministic, CPU-only A2ER equation and official-core audit. No training."""
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)


def literal(delta, q, rho=0.5, dtype=np.float64):
    d=np.asarray(delta,dtype=dtype);q=np.asarray(q,dtype=dtype);r=np.asarray(rho,dtype=dtype)
    with np.errstate(all='ignore'):
        upper=q/r
        eta=1-(np.clip(d,q,upper)-q)/(upper-q)
        corrected=eta*d+(1-eta)*q
        gamma=1-np.sqrt(corrected/d)
    return {'delta':d,'q':q,'eta':eta,'gamma':gamma,'corrected':corrected}


def official_numpy_reference(delta,qold,n=512,eps=np.finfo(np.float32).eps,rho=0.5,lam=0.5,old_priority=1.0):
    d=np.asarray(delta,dtype=np.float64);stable=d+eps
    q=(1-len(d)/n)*qold+len(d)/n*np.quantile(stable,rho,method='linear')
    upper=q/rho;diff=stable-q
    eta=np.clip(1-np.maximum(diff,0)/max(upper-q,np.finfo(np.float32).tiny),0,1)
    ratio=(eta*stable+(1-eta)*q)/stable
    gamma=1-np.sqrt(ratio)
    return {'q':q,'eta':eta,'ratio':ratio,'gamma':gamma,'priority':(1-lam)*old_priority+lam*(1-gamma),'alpha_drive':np.mean(eta*diff)}


def load_official_core():
    namespace={'torch':torch,'nn':nn,'np':np}
    files=[('module/functional.py',{'squareplus','squmoid','squgit'},set()),
           ('derpp/replay_buffer.py',set(),{'ReplayBuffer'}),
           ('derpp/derpp.py',set(),{'DERPP'})]
    for path,functions,classes in files:
        source=ROOT/'sources/a2er'/path;tree=ast.parse(source.read_text());nodes=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
            if isinstance(node,ast.ClassDef) and node.name in classes:
                # Exclude unused persistence/pickle entry points; numeric methods
                # retain their exact original AST and file line numbers.
                node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name not in {'release','load'}]
                nodes.append(node)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


CORE=load_official_core()


class FixedVisitHarness:
    """Controlled visits to 32 entries. Not a simulation of adaptive sampling."""
    def __init__(self,n=512,eps=None,qold=0.0,dtype=torch.float32):
        self.n=n;self.dtype=dtype
        self.engine=CORE['DERPP'](keys=['x'],keys_der=['der'],capacity_stream=32,capacity_reservoir=n,batch_size=32,alpha=1.0,beta=0.5,threshold=0.5,priority=0.5,replay='disjoint',eps=eps,verbose=False)
        self.engine._threshold['der']=torch.tensor(qold,dtype=dtype)
        self.buffer=self.engine.buffer_reservoir[0]
        for i in range(n):self.buffer.append(x=np.array([i],dtype=np.float32),der=np.zeros(2,dtype=np.float32),priority=1.0)
        for i in range(32):self.engine.buffer_stream.append(x=np.array([i],dtype=np.float32))
        self.buffer.fn_select=lambda count:np.arange(count)
        self.engine.buffer_stream.fn_select=lambda count:np.arange(count)

    def step(self,outputs,gradient_check=False):
        h=torch.as_tensor(outputs,dtype=self.dtype,device='cpu').clone().requires_grad_(True)
        zbefore=np.stack(self.buffer.buffer['der'][:32].tolist()).copy()
        priorities_before=self.buffer.priority.copy()
        qbefore=float(self.engine._threshold['der'])
        d=((h.detach().double().numpy()-zbefore.astype(np.float64))**2).sum(-1)/2
        def der_func(x,**kwargs):return {'der':h[x[:,0].to(dtype=torch.int64)]}
        def loss_func(x,**kwargs):return h.sum()*0.0
        total=self.engine(loss_func,der_func)
        gamma=self.engine._gamma['der'].detach().double().numpy()
        q=float(self.engine._threshold['der'])
        zafter=np.stack(self.buffer.buffer['der'][:32].tolist()).copy()
        pr=self.buffer.priority.copy()
        reference=official_numpy_reference(d,qbefore,n=self.n,eps=self.engine.eps)
        record={'n_active':self.n,'n_visited':32,'q_before':qbefore,'q_after':q,'eps':float(self.engine.eps),'dtype':str(self.dtype),
                'delta_before':d,'gamma':gamma,'target_before':zbefore,'target_after':zafter,
                'selected_priority_before':priorities_before[:32],'selected_priority_after':pr[:32],
                'unselected_priorities_unchanged':bool(np.array_equal(pr[32:],priorities_before[32:])),
                'expected_q':reference['q'],'reference_gamma_max_abs':float(np.max(np.abs(gamma-reference['gamma']))),
                'delta_after':((h.detach().double().numpy()-zafter.astype(np.float64))**2).sum(-1)/2,
                'total_objective':float(total.detach()),'all_finite':bool(np.isfinite(gamma).all() and np.isfinite(pr).all())}
        if gradient_check:
            total.backward()
            grad=h.grad.detach().double().numpy()
            expected=reference['ratio'][:,None]*(h.detach().double().numpy()-zbefore)/32
            record['gradient_max_abs_error_to_ratio_detached']=float(np.max(np.abs(grad-expected)))
            record['alpha_real_gradient']=float(self.engine.alpha_real_space['der'].grad)
            record['alpha_real_expected_gradient']=-reference['alpha_drive']
            record['beta_real_gradient']=float(self.engine.beta_real_space.grad)
            corrected_target_gradient=(h.detach().double().numpy()-zafter)/32
            record['gradient_difference_to_detached_corrected_target']=float(np.max(np.abs(grad-corrected_target_gradient)))
            # No optimizer is created or stepped.
        return record


def outputs_for(delta):
    delta=np.asarray(delta,dtype=np.float64)
    return np.stack([np.sqrt(2*delta),np.zeros_like(delta)],axis=1)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return 'NaN' if math.isnan(value) else ('+Inf' if value>0 else '-Inf')
    return value


def main():
    started=time.perf_counter();out=ROOT/'results';out.mkdir(exist_ok=False)
    rows=[];checks=[]
    def check(name,ok,**details):checks.append({'name':name,'passed':bool(ok),**details})
    # Literal domains and positive-domain identities.
    for q in [0.0,1.0]:
        for d in [0.0,1e-12,0.5,1.0,1.5,2.0,4.0]:
            r=literal([d],q);rows.append({'kind':'literal','input_delta':d,'input_q':q,**r})
            if q>0 and d>0:
                g=float(r['gamma'][0]);eta=float(r['eta'][0]);actual=((1-g)*math.sqrt(2*d))**2/2
                check(f'literal_identity_q{q}_d{d}',np.isclose(actual,r['corrected'][0],rtol=1e-12,atol=1e-15),actual=actual,expected=float(r['corrected'][0]))
                check(f'literal_range_q{q}_d{d}',0<=g<=1 and 0<=eta<=1)
    # Prespecified official implementation boundary/control cases.
    cases=[('zero_start',np.zeros(32),0.,512,None),
           ('zero_settled',np.zeros(32),np.finfo(np.float32).eps,512,None),
           ('constant_start',np.ones(32),0.,512,None),
           ('constant_settled',np.ones(32),1.+np.finfo(np.float32).eps,512,None),
           ('constant_partial_buffer',np.ones(32),0.,64,None),
           ('zero_example_eps',np.zeros(32),0.,512,1e-5),
           ('heterogeneous',np.array([0.5]*24+[8.]*8),0.5,512,None)]
    for label,d,q,n,eps in cases:
        engine=FixedVisitHarness(n=n,eps=eps,qold=q)
        r=engine.step(outputs_for(d),gradient_check=(label=='heterogeneous'))
        rows.append({'kind':'official_boundary','case':label,**r})
        check('finite_'+label,r['all_finite'])
        check('reference_'+label,r['reference_gamma_max_abs']<2e-6,error=r['reference_gamma_max_abs'])
        check('unvisited_'+label,r['unselected_priorities_unchanged'])
        if label=='heterogeneous':
            check('official_ratio_gradient',r['gradient_max_abs_error_to_ratio_detached']<2e-7,error=r['gradient_max_abs_error_to_ratio_detached'])
            check('official_alpha_detach_gradient',abs(r['alpha_real_gradient']-r['alpha_real_expected_gradient'])<2e-7)
    # Uniform scale and fixed-epsilon distinction. Scales are of squared norms.
    base=np.array([.5]*24+[8.]*8);base_lit=literal(base,.5)
    for scale in [1e-12,1e-6,1.,1e6,1e12]:
        r=literal(base*scale,.5*scale)
        err=float(np.max(np.abs(r['gamma']-base_lit['gamma'])))
        check('literal_scale_'+str(scale),err<1e-12,error=err)
        official=FixedVisitHarness(qold=.5*scale).step(outputs_for(base*scale))
        rows.append({'kind':'scale','squared_norm_scale':scale,'literal_gamma_max_difference':err,'official':official})
    # Actual target recursion under fixed visits; both successful adaptation and
    # persistent alternating-view constructions retained, not cherry-picked.
    for mode in ['constant_output','alternating_output']:
        engine=FixedVisitHarness(qold=.5)
        for step in range(32):
            h=outputs_for(base)
            if mode=='alternating_output' and step%2:h=-h
            r=engine.step(h)
            rows.append({'kind':'trajectory','mode':mode,'step':step,**r})
            check(f'trajectory_finite_{mode}_{step}',r['all_finite'])
    # Float precision grid, with unsupported torch quantile recorded explicitly.
    for dtype in [np.float16,np.float32,np.float64]:
        for q,d in [(0.,0.),(1e-40,2e-40),(1e-12,2e-12),(1.,1.+1e-7),(1.,1.5),(1e30,2e30)]:
            with np.errstate(all='ignore'):r=literal([d],q,dtype=dtype)
            rows.append({'kind':'precision_literal','dtype':np.dtype(dtype).name,'input_q':q,'input_delta':d,**r})
    for dtype in [torch.float16,torch.float32,torch.float64]:
        try:
            r=FixedVisitHarness(qold=.5,dtype=dtype).step(outputs_for(base))
            rows.append({'kind':'precision_official','dtype':str(dtype),'result':r})
        except (RuntimeError,TypeError,ValueError) as e:
            rows.append({'kind':'precision_official','dtype':str(dtype),'error':str(e),'status':'unsupported_or_failed_without_repair'})
    # No epsilon is silently added to the literal branch. Code initialization is
    # checked separately from any scientific interpretation.
    initial=next(r for r in rows if r.get('case')=='zero_start')
    settled=next(r for r in rows if r.get('case')=='zero_settled')
    check('zero_start_code_gamma_three_quarters',np.allclose(initial['gamma'],.75))
    check('zero_start_code_priority_five_eighths',np.allclose(initial['selected_priority_after'],.625))
    check('zero_settled_code_gamma_zero',np.allclose(settled['gamma'],0.))
    check('zero_start_targets_unchanged',np.array_equal(initial['target_before'],initial['target_after']))
    # Explicit per-feature mean versus half squared L2 sum (Mammoth versus A2ER).
    features=np.arange(1,201,dtype=np.float64).reshape(2,100)/100
    eq_delta=np.sum(features**2,axis=1)/2
    mean_mse=np.mean(features**2)
    check('delta_sum_vs_element_mse',np.isclose(eq_delta.mean(),100/2*mean_mse))
    with (out/'all-cases.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
    (out/'checks.json').write_text(json.dumps(clean(checks),indent=2,allow_nan=False)+'\n')
    summary={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-started,'case_records':len(rows),'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failed':[c for c in checks if not c['passed']],
             'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','optimizer_steps':0,'model_training':False,
             'official_a2er_commit':'f1334df1104724bb3b56f1f353857d92b1141565','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'literal_zero_denominators':'NaN preserved; official code stabilized separately','zero_initial_code_gamma':float(initial['gamma'][0]),'zero_initial_selected_priority':float(initial['selected_priority_after'][0]),
             'scope':'Known deterministic input vectors and fixed visits; not augmentation experiments, full benchmark reproduction, or generalization evidence'}
    (out/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean(summary),indent=2,allow_nan=False))
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':main()
