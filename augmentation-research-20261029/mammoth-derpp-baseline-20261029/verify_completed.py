"""Independent saved-run audit. Does not run a model or access any test data."""
import datetime,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run-002'
PROJECT=ROOT.parents[1]
DATA=PROJECT/'data/cifar100-official-20261029'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    summary=json.loads((RUN/'summary.json').read_text())
    assert summary['status'].startswith('completed_') and summary['optimizer_hook_steps']==6300
    for freeze_name in ['protocol-freeze.json','protocol-freeze-v2.json']:
        freeze=json.loads((ROOT/freeze_name).read_text())
        assert all(sha(ROOT/n)==h for n,h in freeze['files'].items())
    plan=np.load(RUN/'data-and-order-plan.npz',allow_pickle=False)
    train=np.memmap(DATA/'train-only/train.bin',mode='r',dtype=np.uint8,shape=(50000,3074));labels=np.asarray(train[:,1])
    assert not np.intersect1d(np.r_[plan['old_fit'],plan['new_fit']],np.r_[plan['old_dev'],plan['new_dev']]).size
    steps=[json.loads(x) for x in (RUN/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==6300
    rng=np.random.RandomState(20261029)
    buffer_rows=np.full(500,-1,np.int64);arrivals=np.full(500,-1,np.int64);seen=0
    counts=np.zeros((3,50000),np.int64);objective_error=0.;sync_time=0.
    for i,row in enumerate(steps):
        task=i//3150;within=i%3150;epoch=within//63;batch=within%63
        assert (row['task'],row['epoch'],row['batch'],row['global_step'])==(task+1,epoch+1,batch+1,i+1)
        expected_lr=.03 if epoch<35 else .006 if epoch<45 else .0012
        assert abs(row['lr']-expected_lr)<1e-12
        fit=plan['old_fit'] if task==0 else plan['new_fit']
        expected_rows=fit[plan['permutations'][task,epoch,batch*32:(batch+1)*32]]
        assert np.array_equal(expected_rows,row['current_rows'])
        assert row['current_n']==len(expected_rows)==(16 if batch==62 else 32)
        assert np.all((labels[expected_rows]>=task*5)&(labels[expected_rows]<task*5+5))
        np.add.at(counts[0],expected_rows,1)
        expected_draws=0 if seen==0 else 2
        assert len(row['replay_draws'])==expected_draws
        for k,draw in enumerate(row['replay_draws']):
            slots=rng.choice(min(seen,500),size=min(32,seen,500),replace=False)
            assert np.array_equal(slots,draw['slots'])
            assert np.array_equal(buffer_rows[slots],draw['source_rows'])
            assert np.array_equal(arrivals[slots],draw['arrival_indices'])
            assert len(np.unique(slots))==len(slots)
            np.add.at(counts[k+1],buffer_rows[slots],1)
        admissions=[]
        for src in expected_rows:
            slot=seen if seen<500 else rng.randint(0,seen+1)
            seen+=1
            if slot<500:
                buffer_rows[slot]=src;arrivals[slot]=seen-1
                admissions.append({'slot':int(slot),'row':int(src),'arrival_index':seen-1})
        assert admissions==row['reservoir_admitted'] and row['buffer_seen']==seen
        reconstructed=row['ce_current']+.5*row['ce_replay']+.3*row['mse']
        error=abs(reconstructed-row['loss']);objective_error=max(objective_error,error)
        assert error<1e-4*max(1,abs(row['loss']))
        assert math.isfinite(row['loss']) and row['synchronized_wall_seconds']>0
        sync_time+=row['synchronized_wall_seconds']
    assert seen==200000 and counts[0].sum()==200000 and counts[1:].sum()==403136
    assert np.all(counts[0,np.r_[plan['old_fit'],plan['new_fit']]]==50)
    saved=np.load(RUN/'exposure-counts.npz',allow_pickle=False)
    assert np.array_equal(counts[0],saved['current'])
    assert np.array_equal(counts[1],saved['replay_distillation'])
    assert np.array_equal(counts[2],saved['replay_supervised'])
    # Self-created local checkpoints only, their provenance is this frozen run.
    checkpoint_details={}
    for task in [1,2]:
        c=torch.load(RUN/f'task{task}-final.pt',map_location='cpu',weights_only=False)
        assert c['optimizer_hook_steps']==task*3150 and c['completed_epoch']==50
        expected_bn=3*(task*3150)-2
        bn={k:int(v) for k,v in c['model'].items() if k.endswith('num_batches_tracked')}
        assert len(bn)==20 and all(x==expected_bn for x in bn.values())
        group=c['optimizer']['param_groups'][0]
        assert abs(group['lr']-.0012)<1e-12 and group['momentum']==0 and group['weight_decay']==0
        assert c['scheduler']['last_epoch']==50 and c['buffer']['seen']==task*100000
        assert c['model']['classifier.weight'].shape==(10,512)
        rows=c['buffer']['rows'];raw=torch.from_numpy(np.array(train[rows,2:],copy=True).reshape(-1,3,32,32)).float()/255
        assert torch.equal(raw,c['buffer']['examples'])
        assert np.array_equal(labels[rows],c['buffer']['labels'].numpy())
        assert torch.isfinite(c['buffer']['logits']).all()
        if task==2:
            assert np.array_equal(rows,buffer_rows) and np.array_equal(c['buffer']['arrivals'],arrivals)
            state=c['numpy_rng_state'];replayed=rng.get_state()
            assert state[0]==replayed[0] and np.array_equal(state[1],replayed[1]) and state[2:]==replayed[2:]
        checkpoint_details[str(task)]={'steps':c['optimizer_hook_steps'],'BN_updates_per_layer':expected_bn,'buffer_seen':c['buffer']['seen'],'source_image_label_match':True,'historical_targets_finite':True,'last_lr':group['lr']}
        del c
    metrics={}
    for filename,devkey in [('initial-old-dev.npz','old_dev'),('task1-old-dev.npz','old_dev'),('task2-old-dev.npz','old_dev'),('task2-new-dev.npz','new_dev')]:
        data=np.load(RUN/filename,allow_pickle=False);y=labels[plan[devkey]].astype(int);h=data['logits'].astype(np.float64)
        assert np.array_equal(y,data['labels'])
        m=h.max(1);nll=m+np.log(np.exp(h-m[:,None]).sum(1))-h[np.arange(len(y)),y]
        error=(h.argmax(1)!=y)
        assert np.max(np.abs(nll-data['nll']))<1e-12 and np.array_equal(error,data['error'])
        metrics[filename]={'n':len(y),'nll':float(nll.mean()),'accuracy':float(1-error.mean()),'max_saved_NLL_error':float(np.max(np.abs(nll-data['nll'])))}
    p=json.loads((ROOT/'process-timing-v2.json').read_text())
    assert p['exit_code']==0 and not p['timeout_terminated']
    assert p['whole_child_perf_seconds']>=summary['run_perf_seconds']>=sync_time
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_passed':True,'optimizer_steps':6300,'epochs':100,'current_observations':int(counts[0].sum()),'replay_observations':int(counts[1:].sum()),'every_fit_row_seen_50_times':True,'independent_uniform_sampling_and_reservoir_PRNG_replay_exact':True,'final_buffer_rows_arrivals_and_rng_exact':True,'loss_reconstruction_max_abs_error':objective_error,'sum_synchronized_step_seconds':sync_time,'checkpoint_checks':checkpoint_details,'independent_metrics':metrics,'new_model_forwards':0,'new_optimizer_steps':0}
    with (RUN/'independent-verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
