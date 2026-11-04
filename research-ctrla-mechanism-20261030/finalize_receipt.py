from ctrla_adapter import *
from datetime import datetime,timezone
import subprocess

def main():
    status=json.loads((ROOT/'frozen-stage-status.json').read_text());assert status['status']=='complete'
    budget=json.loads((ROOT/'budget-state.json').read_text())
    preserve=json.loads((ROOT/'preservation-after.json').read_text())
    assert not preserve['changed_historical_tracked_files'] and not preserve['changed_upstream_source_files']
    seeds=[json.loads((ROOT/f'seed{s}-frozen-results.json').read_text()) for s in (0,1)]
    for seed in (0,1):
        row=json.loads((ROOT/f'seed{seed}-latest-checkpoint.json').read_text())
        assert row['actual_epoch']==10 and row['phase_update_done'] and sha256(ROOT/row['path'])==row['sha256']
    ignored=['data/train_images.npy','weights/seed0-epoch10.pt','raw/seed0-audit-raw-N5-S-full.npz','logs/training-events.jsonl','sources/ControlAugment.tar.gz']
    receipt=subprocess.check_output(['git','check-ignore','-v',*[str(ROOT/p) for p in ignored]],cwd=ROOT.parent,text=True)
    summary={'finalized_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','decision':status['decision'],'seeds':[{ 'seed':r['seed'],'primary_gate':r['resource_gate'],'renderer_rematched_audit_CE_acceptance':{k:v['audit_CE_match']['accepted'] for k,v in r['families']['renderer_rematched']['renderers'].items()},'frozen_model_BN_unchanged':r['frozen_model_and_BN_bitwise_unchanged']} for r in seeds],'budget':budget,'remaining_original_planning_ceiling':{'train_images':9200000-budget['train_images'],'original_46k_exposures':200-budget['train_images']/46000,'inference_image_forwards':8000000-budget['inference_images'],'authorization_for_future_stage':False},'tests':{'adapter':10,'rendering_geometry_subset':7,'data':5,'statistics':4,'spawn_smoke':True,'reader_tensor_equivalence':True},'historical_tracked_unchanged':2066,'official_source_unchanged':True,'local_ignore_evidence':receipt,'blockers':[],'scientific_limits':['shortened pilot; not v2 or final accuracy reproduction','small positive rendering benefit below resource gate, not a zero-effect finding','seed1 rematched CE confidence intervals fail predeclared +/-5% containment; no exclusion of total difficulty','bootstrap quantifies original-image uncertainty, not training-seed uncertainty','online feedback compensation untested; no online fork']}
    write_json(ROOT/'execution-summary.json',summary)
    index=['# Ctrl-A mechanism diagnostic — local archive','',f"Outcome: **{status['decision']}**. Two seeds and all pre-registered frozen measurements completed. No online branches or Colab usage.",'',
           '- [Chinese results and limits](RESULTS_ZH.md)','- [Machine-readable outcome, budget and remaining ceiling](execution-summary.json)','- [Protocol](protocol.json) · [pre-outcome lock](protocol-lock.json) · [sampling manifest](sampling-manifest.json)','- [Source provenance](source-manifest.json) · [CIFAR asset hashes](asset-manifest.json) · [interface patch manifest](patch-manifest.json)','- [Compatibility/layout recovery](compatibility-amendment-layout.json) · [spawn recovery](recovery-spawn.json) · [reader equivalence](reader-benchmark.json)','- [Checkpoint continuation semantics](CHECKPOINT_HANDOFF.md)','- [Aggregate metrics](aggregate-metrics.csv) · [all operation strata and CIs](operation-metrics.csv) · [official ROR and CIs](official-ror-metrics.csv)','- [Seed0 full results](seed0-frozen-results.json) · [seed1 full results](seed1-frozen-results.json)','- [19 pre-training tests](tests/all-pretrain-tests-recovery.txt) · [result invariance](result-integrity.json) · [historical preservation](preservation-after.json)','- [Paired rendering effects](figures/paired-rendering-gain.png) · [training health](figures/training-health.png) · [official ROR](figures/official-ror.png)','',
           'All raw predictions, per-image CE, model/BN and optimizer/RNG states, and full timing logs remain in local ignored raw/, weights/ and logs/. No test batch was extracted or loaded. No commit or upload was performed. Final SHA256 values are in deliverable-manifest.json.']
    (ROOT/'README.md').write_text('\n'.join(index)+'\n')
    files={str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ('sources','data','__pycache__')) and p.name!='deliverable-manifest.json'}
    write_json(ROOT/'deliverable-manifest.json',{'finalized_utc':summary['finalized_utc'],'files':files,'decision':status['decision'],'budget':budget,'sources_and_data_hashes':'source-manifest.json and asset-manifest.json','no_active_writers':True})
    print(json.dumps({'decision':status['decision'],'files_hashed':len(files),'train_images':budget['train_images'],'inference_images':budget['inference_images'],'elapsed_minutes':budget['elapsed_wall_seconds']/60,'remaining_planning_budget':summary['remaining_original_planning_ceiling'],'blockers':[]},indent=2))

if __name__=='__main__':main()
