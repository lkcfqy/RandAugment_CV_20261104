"""Pre-registered frozen challenges, CE matching, paired image-cluster inference."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_FAST_MATH']='0'
import time,traceback,csv
from ctrla_adapter import *
from train_common import loader,predict_loader,ror_eval,event
from torch.utils.data import Dataset

class ChallengeDataset(Dataset):
    def __init__(self,role,gamma,alpha,coefficient,renderer,component='full',depth=5):
        self.indices=np.load(ROOT/'data'/f'{role}_indices.npy')
        raw=np.load(ROOT/'data'/f'{role}_challenges.npz')
        self.order=raw['order'];self.flip=raw['flip'];self.crop=raw['crop']
        self.mags=strengths(self.order,raw['quantiles'],raw['signs'],gamma,alpha,coefficient)
        self.renderer=renderer;self.component=component;self.depth=depth;self.images=None;self.labels=None
    def __len__(self):return len(self.indices)*(1 if self.component=='base' else 15)
    def __getitem__(self,index):
        if self.images is None:self.images=np.load(ROOT/'data'/'train_images.npy',mmap_mode='r');self.labels=np.load(ROOT/'data'/'train_labels.npy',mmap_mode='r')
        if self.component=='base':local=index;op=0
        else:local,op=divmod(index,15)
        original=self.indices[local];img=base_image(self.images[original],self.flip[local],self.crop[local])
        order=self.order[local,op];mags=self.mags[local,op]
        if self.depth==2:order=order[-2:];mags=mags[-2:]
        if self.component=='prefix':order=order[:-1];mags=mags[:-1]
        elif self.component=='single':order=order[-1:];mags=mags[-1:]
        if self.component!='base':img=render(img,order,mags,self.renderer)
        return tensor_image(img),int(self.labels[original])

def evaluate(model,state,budget,seed,role,coefficient,renderer,component='full',depth=5,tag=''):
    t=time.monotonic();ds=ChallengeDataset(role,state['Gamma'],state['alpha'],coefficient,renderer,component,depth)
    # Reader-only benchmark proved bitwise identical tensors/order with workers0;
    # avoid the measured 35s spawn/teardown overhead per frozen challenge pass.
    pred,loss,y=predict_loader(model,loader(ds,workers=0),state['mean'],state['std'],budget)
    n=len(ds.indices);width=1 if component=='base' else 15
    out={'prediction':pred.reshape(n,width),'ce':loss.reshape(n,width),'target':y.reshape(n,width),'coefficient':np.array(coefficient)}
    name=f'seed{seed}-{role}-{tag}-N{depth}-{renderer}-{component}.npz'
    np.savez_compressed(ROOT/'raw'/name,**out)
    event('frozen_view_complete',seed=seed,role=role,tag=tag,depth=depth,renderer=renderer,component=component,coefficient=coefficient,images=len(ds),ce=float(loss.mean()),accuracy=float((pred==y).mean()),elapsed_seconds=time.monotonic()-t)
    return out

def match_strength(model,state,budget,seed):
    target=evaluate(model,state,budget,seed,'strength_match',1.,'S',depth=2,tag='reference')
    target_ce=float(target['ce'].mean());result={'target_N2_S_ce':target_ce,'renderers':{}}
    for renderer in 'SEF':
        trials={}
        def trial(coefficient):
            if coefficient not in trials:
                out=evaluate(model,state,budget,seed,'strength_match',coefficient,renderer,tag=f'match-{coefficient:.9f}')
                trials[coefficient]=float(out['ce'].mean())
            return trials[coefficient]-target_ce
        grid=[i/10 for i in range(11)]
        for c in grid:trial(c)
        brackets=[(a,b) for a,b in zip(grid[:-1],grid[1:]) if trial(a)*trial(b)<=0]
        feasible=bool(brackets)
        if brackets:
            a,b=min(brackets,key=lambda pair:(min(abs(trial(pair[0])),abs(trial(pair[1]))),pair[0]))
            for _ in range(4):
                c=(a+b)/2;value=trial(c)
                if trial(a)*value<=0:b=c
                else:a=c
        chosen=min(trials,key=lambda c:(abs(trials[c]-target_ce),c))
        result['renderers'][renderer]={'coefficient':chosen,'mean_ce':trials[chosen],'relative_ce_difference':trials[chosen]/target_ce-1,'target_bracket_found':feasible,'trials':[{'coefficient':c,'mean_ce':v} for c,v in sorted(trials.items())]}
        write_json(ROOT/f'seed{seed}-strength-match.json',result)
        event('match_chosen',seed=seed,renderer=renderer,**{k:v for k,v in result['renderers'][renderer].items() if k!='trials'})
    return result

def ci(values):
    values=np.asarray(values);finite=values[np.isfinite(values)]
    if not len(finite):return None
    return [float(v) for v in np.quantile(finite,[.025,.975])]

def estimate(point,draws):return {'estimate':float(point),'ci95':ci(draws)}

def bootstrap_weights(n):
    rng=np.random.default_rng(20261032)
    weights=np.empty((2000,n),np.float64)
    for j in range(2000):weights[j]=np.bincount(rng.integers(0,n,n),minlength=n)/n
    return weights

def err(out):return (out['prediction']!=out['target']).astype(np.float64)

def mechanism_metrics(base,single,prefix,full,w):
    eb=err(base);ei=err(single);ep=err(prefix);ef=err(full)
    interaction=ef-ep-ei+eb
    new_wrong=(ep==0)&(ef==1);corrected=(ep==1)&(ef==0)
    ap=1-ep.mean(0);af=1-ef.mean(0);ab=1-eb.mean(0);ai=1-ei.mean(0)
    bap=1-w@ep;baf=1-w@ef;bab=1-w@eb;bai=1-w@ei
    with np.errstate(divide='ignore',invalid='ignore'):
        conditional=af/ap;single_r=ai/ab;gap=conditional-single_r
        bconditional=baf/bap;bsingle=bai/bab;bgap=bconditional-bsingle
    rows=[]
    for op in range(15):
        valid=bool(ap[op]>=.10 and (1-ep[:,op]).sum()>=100 and ab[0]>=.10 and (1-eb[:,0]).sum()>=100)
        row={'operation':OPS[op],'weight':1/15,'prefix_accuracy':float(ap[op]),'full_accuracy':float(af[op]),'base_accuracy':float(ab[0]),'single_accuracy':float(ai[op]),'ratio_valid':valid,'E':estimate(interaction[:,op].mean(),w@interaction[:,op]),'new_wrong':estimate(new_wrong[:,op].mean(),w@new_wrong[:,op]),'corrected_wrong':estimate(corrected[:,op].mean(),w@corrected[:,op]),'conditional_R':estimate(conditional[op],bconditional[:,op]) if valid else None,'single_R':estimate(single_r[op],bsingle[:,op]) if valid else None,'ROR_gap':estimate(gap[op],bgap[:,op]) if valid else None}
        rows.append(row)
    allvalid=all(row['ratio_valid'] for row in rows)
    aggregate={'E':estimate(interaction.mean(),w@interaction.mean(1)),'new_wrong':estimate(new_wrong.mean(),w@new_wrong.mean(1)),'corrected_wrong':estimate(corrected.mean(),w@corrected.mean(1)),'error':estimate(ef.mean(),w@ef.mean(1)),'all_ROR_strata_valid':allvalid,'uniform_weighted_ROR_gap':estimate(gap.mean(),bgap.mean(1)) if allvalid else None}
    return {'aggregate':aggregate,'by_operation':rows}

def audit_matching(reference,full,w):
    a=full['ce'].mean(1);b=reference['ce'].mean(1)
    draws=(w@a)/(w@b)-1;point=a.mean()/b.mean()-1;bounds=ci(draws)
    return {**estimate(point,draws),'accepted':bool(abs(point)<=.05 and bounds[0]>=-.05 and bounds[1]<=.05),'criterion':'point AND complete95%CI within +/-5%','mean_ce':float(a.mean()),'N2_S_mean_ce':float(b.mean())}

def compare_renderers(s,e,f,w):
    result={}
    for name,other in [('S_minus_E',e),('S_minus_F',f),('E_minus_F',None)]:
        diff=err(e)-err(f) if other is None else err(s)-err(other)
        result[name]={'aggregate':estimate(diff.mean(),w@diff.mean(1)),'by_operation':[{ 'operation':OPS[i],**estimate(diff[:,i].mean(),w@diff[:,i])} for i in range(15)]}
    return result

def diagnose_seed(seed,budget):
    if (ROOT/f'seed{seed}-frozen-results.json').exists():raise RuntimeError('Completed frozen seed exists; refuse rerun')
    state=torch.load(ROOT/'weights'/f'seed{seed}-epoch10.pt',map_location='cpu',weights_only=False)
    assert state['actual_epoch']==10 and state['phase_update_done']
    model=make_model();model.load_state_dict(state['model']);model.eval()
    for param in model.parameters():param.requires_grad_(False)
    matching=match_strength(model,state,budget,seed)
    base=evaluate(model,state,budget,seed,'audit',1.,'S','base',tag='common')
    n2={r:evaluate(model,state,budget,seed,'audit',1.,r,depth=2,tag='reference') for r in 'SEF'}
    raw={};shared={};rematched={};c_s=matching['renderers']['S']['coefficient']
    for family,coefficient in [('raw',1.),('shared',c_s)]:
        target=raw if family=='raw' else shared
        for renderer in 'SEF':
            target[renderer]={comp:evaluate(model,state,budget,seed,'audit',coefficient,renderer,comp,tag=family) for comp in ('single','prefix','full')}
    rematched['S']=shared['S']
    for renderer in 'EF':
        coefficient=matching['renderers'][renderer]['coefficient']
        rematched[renderer]={comp:evaluate(model,state,budget,seed,'audit',coefficient,renderer,comp,tag='rematched') for comp in ('single','prefix','full')}
    ror_eval(model,state['mean'],state['std'],budget,seed,10,role='audit')
    unchanged=all(torch.equal(value.detach().cpu(),state['model'][key]) for key,value in model.state_dict().items())
    if not unchanged:raise RuntimeError('Frozen model/BN state changed during diagnostics')
    w=bootstrap_weights(2000)
    result={'seed':seed,'checkpoint_sha256':sha256(ROOT/'weights'/f'seed{seed}-epoch10.pt'),'status':'frozen_complete','matching_selection':matching,'bootstrap':{'clusters':'original audit image, retain all15 terminal-op contexts','replicates':2000,'training_seed_uncertainty':False},'N2':{r:{'error':estimate(err(v).mean(),w@err(v).mean(1)),'mean_ce':float(v['ce'].mean())} for r,v in n2.items()},'families':{}}
    for name,family in [('raw',raw),('same_parameter_matched',shared),('renderer_rematched',rematched)]:
        result['families'][name]={'renderers':{r:{'coefficient':float(v['full']['coefficient']),'audit_CE_match':audit_matching(n2['S'],v['full'],w),'mechanism':mechanism_metrics(base,v['single'],v['prefix'],v['full'],w)} for r,v in family.items()},'paired_rendering_differences':compare_renderers(family['S']['full'],family['E']['full'],family['F']['full'],w)}
    gate=result['families']['raw']['paired_rendering_differences']['S_minus_F']['aggregate']
    result['frozen_model_and_BN_bitwise_unchanged']=unchanged
    result['resource_gate']={'contrast':'raw N5 error(S)-error(F); positive=F improves',**gate,'seed_pass':bool(gate['estimate']>=.02 and gate['ci95'][0]>0),'role':'resource gate, not scientific significance standard','online_branch_started':False}
    write_json(ROOT/f'seed{seed}-frozen-results.json',result)
    event('frozen_seed_complete',seed=seed,resource_gate=result['resource_gate'])
    del model;torch.mps.empty_cache();return result

def main():
    torch.set_num_threads(6);budget=Budget()
    assert json.loads((ROOT/'common-stage-status.json').read_text())['status']=='complete'
    try:
        results=[diagnose_seed(seed,budget) for seed in (0,1)]
        passed=all(r['resource_gate']['seed_pass'] for r in results)
        write_json(ROOT/'frozen-stage-status.json',{'status':'complete','seeds':[0,1],'resource_gate_passed':passed,'decision':'SIGNAL_FOR_PARENT_REVIEW_ONLY' if passed else 'NO_GO_FOR_ONLINE_FORK','online_branch_started':False,'feedback_compensation_tested':False})
        budget.save()
    except Exception as e:
        budget.save();write_json(ROOT/'frozen-stage-status.json',{'status':'stopped_indeterminate','error':str(e),'traceback':traceback.format_exc(),'online_branch_started':False});raise

if __name__=='__main__':main()
