"""Verify stored logits and controls using only stdlib; never rerun training/inference."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    result=json.loads((ROOT/'diagnostic-results.json').read_text())
    old=json.loads((BASE/'stage-validity-pilot-20261021/finetune_TA-result.json').read_text())
    labels=[r['Y'] for r in json.loads((BASE/'stage-validity-pilot-20261021/protocol.json').read_text())['train_rows']]
    count=0;max_loss_error=0.;max_mean_error=0.;total_compute=0.;initial=None
    for arm in protocol['arms']:
        name=arm['name'];a=json.loads((ROOT/(name+'-result.json')).read_text())
        assert a['initial_state_sha256']==old['initial_state_sha256']
        assert a['exact_training_RGB_order_sha256']==old['exact_training_RGB_order_sha256']
        assert a['freeze_BN_affine']==arm['freeze_BN_affine'] and a['BN_running_stats_unchanged']
        assert (a['BN_affine_initial_sha256']==a['BN_affine_final_sha256'])==arm['freeze_BN_affine']
        assert a['conv1_delta_L2']>0 and a['parameter_devices']==['mps'] and a['all_loss_grad_parameter_finite']
        assert a['steps']==128 and a['training_presentations']==1024
        trace=[json.loads(line) for line in (ROOT/(name+'-steps.jsonl')).read_text().splitlines()]
        assert len(trace)==128 and [r['step'] for r in trace]==list(range(1,129))
        assert all(math.isfinite(r['loss']) and r['guarded_compute_seconds']>0 for r in trace)
        elapsed=sum(r['guarded_compute_seconds'] for r in trace)
        assert abs(elapsed-a['training_compute_seconds'])<1e-10;total_compute+=elapsed
        assert abs(sum(r['loss'] for r in trace[-32:])/32-a['last32_online_TA_mean_loss'])<1e-10
        assert set(a['checkpoints'])=={'0','64','128'}
        if initial is None:initial=a['checkpoints']['0']
        else:assert initial==a['checkpoints']['0']
        for checkpoint in a['checkpoints'].values():
            for m in checkpoint.values():
                assert len(m['logits'])==len(m['losses'])==64
                losses=[];correct=0
                for y,z,saved_loss in zip(labels,m['logits'],m['losses']):
                    assert len(z)==2 and all(math.isfinite(x) for x in z)
                    peak=max(z);ll=peak+math.log(sum(math.exp(x-peak) for x in z))-z[y]
                    losses.append(ll);correct+=int(max(range(2),key=lambda k:z[k])==y)
                    max_loss_error=max(max_loss_error,abs(ll-saved_loss));count+=1
                max_mean_error=max(max_mean_error,abs(sum(losses)/64-m['mean_LL_nats']))
                assert correct/64==m['accuracy']
        final=a['checkpoints']['128']['train_clean']
        assert a['passes_learning_gate']==(final['accuracy']>=.9 and final['mean_LL_nats']<=.2)
        assert a['underlearning_gate']==(final['accuracy']<=.8 or final['mean_LL_nats']>=.5)
        assert not a['passes_learning_gate'] and a['underlearning_gate']
        assert result['final_metrics'][name]['train_clean']==final
    assert count==768 and max_loss_error<1e-5 and max_mean_error<1e-5
    assert abs(total_compute-result['training_compute_seconds'])<1e-10
    assert total_compute<120 and result['total_wall_seconds']<240
    assert result['protocol_sha256']==sha(ROOT/'protocol.json') and result['code_sha256']==sha(ROOT/'diagnostic.py')
    assert result['interpretation']=='affine_freeze_did_not_rescue_learning_at_this_fixed_budget_no_further_tuning'
    assert result['development_test_evaluations']==0 and not result['A100'] and not result['ARIS']
    return dict(status='stored_metrics_and_controls_verified_without_execution',stored_predictions_recomputed=count,
        max_per_image_LL_error=max_loss_error,max_mean_LL_error=max_mean_error,accuracies_exact=True,
        training_steps_verified=256,all_recorded_losses_finite=True,initial_and_training_RGB_match_old=True,
        no_development_or_test_evaluation=True,compute_and_wall_caps_met=True,
        both_arms_failed_predeclared_learning_gate=True,no_training_or_inference_rerun=True)
if __name__=='__main__':
    out=main()
    with (ROOT/'verification.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
