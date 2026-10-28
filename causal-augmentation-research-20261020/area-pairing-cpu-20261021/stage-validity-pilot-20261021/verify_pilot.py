"""Readback of matched development pilot; no new inference, training or hypothesis tests."""
from pathlib import Path
from collections import Counter
import hashlib,json,math

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return json.loads((ROOT/name).read_text())
def main():
    protocol=load('protocol.json');result=load('pilot-results.json');manifest=load('input-manifest.json')
    assert result['protocol_sha256']==sha(ROOT/'protocol.json') and result['code_sha256']==sha(ROOT/'pilot.py')
    assert len(manifest['orders'])==16 and all(sorted(row)==list(range(64)) for row in manifest['orders'])
    assert len(manifest['TA_views'])==1024
    assert Counter(v['source_id'] for v in manifest['TA_views'])=={r['source_id']:16 for r in protocol['train_rows']}
    assert not {r['source_id'] for r in protocol['train_rows']}&{r['source_id'] for r in protocol['development_rows']}
    assert protocol['development_status'].startswith('all64previously used')
    assert result['total_wall_seconds']<600 and result['total_guarded_training_compute_seconds']<600
    error=0.;arms={};equality=None;means={}
    for name in ['frozen_base','finetune_base','frozen_TA','finetune_TA']:
        arm=load(name+'-result.json');arms[name]=arm
        trace=[json.loads(line) for line in (ROOT/(name+'-steps.jsonl')).read_text().splitlines()]
        assert len(trace)==128 and [r['step'] for r in trace]==list(range(1,129))
        assert all(math.isfinite(r['loss']) and r['guarded_compute_seconds']>0 for r in trace)
        assert abs(sum(r['guarded_compute_seconds'] for r in trace)-arm['training_compute_seconds'])<1e-9
        assert arm['steps']==128 and arm['training_presentations']==1024 and arm['BN_running_stats_unchanged']
        assert arm['head_delta_L2']>0 and arm['all_loss_grad_and_parameter_checks_passed']
        if arm['finetune']:assert arm['body_before_sha256']!=arm['body_after_sha256'] and arm['conv1_delta_L2']>0
        else:assert arm['body_before_sha256']==arm['body_after_sha256'] and arm['conv1_delta_L2']==0
        means[name]={}
        for step in ['0','64','128']:
            for group,rows in [('train_clean',protocol['train_rows']),('development_clean',protocol['development_rows'])]:
                prediction=arm['checkpoints'][step][group];losses=[];correct=0
                for z,row,stored_loss in zip(prediction['logits'],rows,prediction['losses']):
                    maximum=max(z);LL=maximum+math.log(math.fsum(math.exp(v-maximum) for v in z))-z[row['Y']]
                    losses.append(LL);error=max(error,abs(LL-stored_loss));correct+=int((z[1]>z[0])==bool(row['Y']))
                mean=math.fsum(losses)/64
                assert abs(mean-prediction['mean_LL_nats'])<1e-5 and abs(correct/64-prediction['accuracy'])<1e-12
                if group=='development_clean':means[name][step]=mean
        if equality is None:equality=arm['checkpoints']['0']
        else:assert arm['checkpoints']['0']==equality
    assert arms['frozen_base']['exact_training_RGB_order_sha256']==arms['finetune_base']['exact_training_RGB_order_sha256']
    assert arms['frozen_TA']['exact_training_RGB_order_sha256']==arms['finetune_TA']['exact_training_RGB_order_sha256']
    for step in ['0','64','128']:
        Gf=means['frozen_base'][step]-means['frozen_TA'][step];Gt=means['finetune_base'][step]-means['finetune_TA'][step]
        for key,val in [('G_frozen',Gf),('G_finetune',Gt),('D',Gt-Gf)]:
            assert abs(val-result['effects_by_fixed_checkpoint'][step][key])<1e-5
    verify=dict(status='passed',steps=512,train_presentations=4096,
        all_initial_clean_predictions_bitwise_identical=True,all_heads_updated=True,
        frozen_backbone_unchanged_and_finetuned_backbone_updated=True,allBNrunningstats_fixed=True,
        exact_source_order_and_RGB_input_match_between_stages=True,per_source_train_budget_exactly16=True,
        independent_math_fsum_per_photo_loss_max_abs_error=error,all_fixed_checkpoint_effects_recomputed=True,
        total_wall_seconds=result['total_wall_seconds'],total_guarded_training_compute_seconds=result['total_guarded_training_compute_seconds'],
        used_development_set_not_independent_test=True,no_new_data=True,no_extra_runs=True,CPU_only=True,GPU=False,ARIS=False)
    with (ROOT/'verification.json').open('x') as f:json.dump(verify,f,indent=2);f.write('\n')
    print(json.dumps(verify,indent=2))

if __name__=='__main__':main()
