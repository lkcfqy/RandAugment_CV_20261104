"""Stdlib verification of existing training outputs, no torch/model execution."""
from pathlib import Path
import collections,hashlib,json,math
ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
def load(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    result=load('recovery-results.json');protocol=load('protocol.json');inputs=load('input-audit.json');probe=load('probe-manifest.json')
    rows=inputs['source_photos'];labels=[r['Y'] for r in rows];ids={r['source_id'] for r in rows};y_map={r['source_id']:r['Y'] for r in rows}
    assert len(rows)==64 and collections.Counter(labels)=={0:32,1:32} and len(probe)==256
    assert inputs['labels_agree_official_species_metadata'] and inputs['base_RGB_all_match_old'] and inputs['fixed_TA_probe_matches_old']
    assert inputs['inverse_normalization_min']>=-1e-6 and inputs['inverse_normalization_max']<=1+1e-6
    old=json.loads((BASE/'stage-validity-pilot-20261021/input-manifest.json').read_text())
    old_views={(r['pass_index'],r['source_id']):r for r in old['TA_views']}
    traces={};arms={};predictions=0;max_loss_error=0.;max_mean_error=0.
    for name in ['base','TA']:
        a=load(name+'-result.json');arms[name]=a
        trace=[json.loads(x) for x in (ROOT/(name+'-steps.jsonl')).read_text().splitlines()];traces[name]=trace
        assert len(trace)==2048 and [r['step'] for r in trace]==list(range(1,2049))
        assert all(r['pass_index']==(r['step']-1)//8 and math.isfinite(r['loss']) and r['guarded_compute_seconds']>0 for r in trace)
        assert abs(sum(r['guarded_compute_seconds'] for r in trace)-a['training_compute_seconds'])<1e-9
        assert a['steps']==2048 and a['presentations']==16384 and a['all_loss_grad_parameter_finite']
        assert a['BN_running_stats_unchanged'] and a['BN_affine_trainable'] and a['actual_parameter_devices']==['mps']
        assert set(a['checkpoints'])=={'0','128','256','512','1024','1536','2048'}
        for b in range(16):assert abs(sum(r['loss'] for r in trace[b*128:(b+1)*128])/128-a['block128_online_mean_LL'][b])<1e-10
        for checkpoint in a['checkpoints'].values():
            for setting,m in checkpoint.items():
                expected_y=labels if setting=='train_clean' else [r['Y'] for r in probe]
                assert len(m['logits'])==len(m['losses'])==len(expected_y)
                recomputed=[];correct=0
                for y,z,saved in zip(expected_y,m['logits'],m['losses']):
                    assert len(z)==2 and all(math.isfinite(x) for x in z)
                    peak=max(z);ll=peak+math.log(sum(math.exp(x-peak) for x in z))-z[y]
                    correct+=int(max(range(2),key=lambda i:z[i])==y);recomputed.append(ll)
                    max_loss_error=max(max_loss_error,abs(ll-saved));predictions+=1
                assert correct/len(expected_y)==m['accuracy']
                max_mean_error=max(max_mean_error,abs(sum(recomputed)/len(recomputed)-m['mean_LL_nats']))
        assert a['checkpoints']['2048']==result['final_metrics'][name]
    assert arms['base']['initial_state_sha256']==arms['TA']['initial_state_sha256']
    assert arms['base']['checkpoints']['0']==arms['TA']['checkpoints']['0']
    assert predictions==4480 and max_loss_error<1e-5 and max_mean_error<1e-5
    views=[json.loads(x) for x in (ROOT/'TA-risk-views.jsonl').read_text().splitlines()];assert len(views)==16384
    operations=collections.Counter();collisions={};per_source=collections.Counter();positions=collections.defaultdict(list)
    old_matches=0
    for r in views:
        source=r['source_id'];epoch=r['pass_index'];assert source in ids and r['Y']==y_map[source] and 0<=epoch<256
        seed=int.from_bytes(hashlib.sha256(('stage-pilot-v1:'+str(epoch)+':'+source).encode()).digest()[:8],'little')&((1<<63)-1)
        assert r['seed']==seed;per_source[source]+=1;positions[epoch].append(source);operations[r['operation']]+=1
        collisions.setdefault(r['RGB_sha256'],[0,0])[r['Y']]+=1
        if epoch<16:
            assert r['RGB_sha256']==old_views[(epoch,source)]['RGB_sha256'];old_matches+=1
        before=r['base_foreground_pixels'];after=r['foreground_pixels']
        assert 0<=before<=224*224 and 0<=after<=224*224
        if before:assert abs(r['foreground_retention']-after/before)<1e-12
        else:assert r['foreground_retention'] is None
        assert max(r['channel_std_RGB255'])==r['max_channel_spatial_std_RGB255']
        assert r['exact_constant']==all(v==0 for v in r['channel_dynamic_ranges_RGB255'])
    assert old_matches==1024 and all(n==256 for n in per_source.values())
    assert all(len(order)==64 and set(order)==ids for order in positions.values())
    old_id_order=[r['source_id'] for r in rows]
    assert all(positions[e]==[old_id_order[i] for i in old['orders'][e]] for e in range(16))
    risk=load('risk-summary.json');assert dict(operations)==risk['operation_counts']
    mixed={h:v for h,v in collisions.items() if all(v)}
    floor=sum(sum(v)*(-sum((n/sum(v))*math.log(n/sum(v)) for n in v if n)) for v in collisions.values())/16384
    assert risk['cross_label_exact_RGB_collision_hashes']==len(mixed) and risk['cross_label_exact_collision_views']==sum(sum(v) for v in mixed.values())
    assert abs(floor-risk['empirical_fixed_library_cross_entropy_lower_bound_nats'])<1e-12
    assert risk['foreground_retention_eq0_count']==sum(r['foreground_retention']==0 for r in views)
    assert risk['foreground_retention_below_0_1_count']==sum(r['foreground_retention'] is not None and r['foreground_retention']<.1 for r in views)
    assert risk['exact_constant_count']==sum(r['exact_constant'] for r in views)
    assert result['main_wall_seconds']<600 and result['training_compute_seconds']<480
    assert abs(sum(a['training_compute_seconds'] for a in arms.values())-result['training_compute_seconds'])<1e-9
    assert result['development_or_test_photos_or_evaluations']==0 and result['no_new_data_or_software'] and not result['A100'] and not result['ARIS']
    assert result['code_sha256']==sha(ROOT/'recovery.py') and result['protocol_sha256']==sha(ROOT/'protocol.json')
    return dict(status='verified_stored_outputs_without_rerunning_models',training_steps=4096,TA_risk_views=16384,
        old_TA_RGB_matches=old_matches,each_source_presented_per_arm=256,stored_predictions_recomputed=predictions,
        max_per_image_LL_error=max_loss_error,max_mean_LL_error=max_mean_error,accuracies_exact=True,
        matched_update_counts=True,all_recorded_losses_finite=True,risk_counts_and_exact_collision_bound_verified=True,
        no_held_out_evaluation=True,compute_wall_caps_met=True)
if __name__=='__main__':
    result=verify()
    with (ROOT/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
