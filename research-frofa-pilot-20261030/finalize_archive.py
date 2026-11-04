"""Read-only audit of completed outputs plus final manifest; no model/math reruns."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib,json,datetime,time,subprocess
ROOT=_portable_project_path(__file__, 'research-frofa-pilot-20261030')
PROJECT=ROOT.parent
def load(name):return json.loads((ROOT/name).read_text())
def digest(path,algo='sha256'):
    h=hashlib.new(algo)
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()
def main():
    ledger=load('ledger.json');result=load('RESULTS.json');p=load('protocol.json');fits=[json.loads(f.read_text()) for f in sorted((ROOT/'fits').glob('*.json'))];test=[json.loads(f.read_text()) for f in sorted((ROOT/'test').glob('*.json'))]
    checks={}
    checks['worker_completed']=ledger['phase']=='completed' and not (ROOT/'STOP.json').exists()
    checks['exact_complete_fit_count']=len(fits)==36 and sum(f['phase']=='calibration' for f in fits)==6 and sum(f['phase']=='confirm' for f in fits)==30
    checks['all_complete_fits_400_updates']=all(f['updates']==400 for f in fits)
    checks['six_full_paired_episodes']=all({f['arm'] for f in fits if f['phase']=='confirm' and f['seed']==seed}==set(p['arms']) and len({(f['initial_sha256'],f['batch_order_sha256'],tuple(f['support_indices'])) for f in fits if f['phase']=='confirm' and f['seed']==seed})==1 for seed in range(1,7))
    rows=load('manifest.json')['rows']
    checks['confirmation_support_excludes_calibration']=all(len(set(f['support_indices']))==100 and all(10<=rows[i]['class_rank']<100 for i in f['support_indices']) and all(sum(rows[i]['label']==c for i in f['support_indices'])==10 for c in range(10)) for f in fits if f['phase']=='confirm')
    checks['one_test_per_frozen_head']=len(test)==30 and all(t['count']==3000 for t in test) and len(list((ROOT/'test').glob('*-pred.npy')))==30
    frozen=load('all-confirmation-models-frozen.json');frozen_epoch=datetime.datetime.fromisoformat(frozen['utc']).timestamp()
    checks['all_heads_sealed_before_test_files']=frozen['count']==30 and frozen['before_any_test_metric'] and min(f.stat().st_mtime for f in (ROOT/'test').glob('*.json'))>=frozen_epoch
    checks['test_values_match_result_table']=all(t['accuracy_percent']==result['accuracy_percent_by_episode'][str(t['seed'])][t['arm']] and abs(t['accuracy_percent']-100*t['correct']/t['count'])<1e-12 for t in test)
    checks['exact_update_and_exposure_counts']=ledger['counts']['updates']==14600 and ledger['counts']['training_exposures']==292000 and ledger['counts']['preflight_updates']==200
    checks['exact_eval_counts']=ledger['counts']['eval_images']==96200 and ledger['counts']['test_eval_images']==90000 and ledger['counts']['calibration_eval_images']==6000
    checks['all_resource_counters_within_caps']=all(ledger['counts'].get(k,0)<=cap for k,cap in p['limits'].items() if k in ledger['counts'])
    loading=load('vision-weight-loading.json')
    checks['vision_weights_complete']=not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs']
    checks['unused_keys_only_text_and_logits']=all(k.startswith('text_model.') or k in ['logit_scale','logit_bias'] for k in loading['unexpected_keys'])
    seal=load('source-seal-resume.json');checks['active_source_seal_unchanged']=all(digest(ROOT/name)==h for name,h in seal.items())
    checks['dataset_md5']=digest(ROOT/'assets/EuroSAT_RGB.zip','md5')==p['assets']['dataset_md5']
    previous=load('preservation-before.json');old=PROJECT/'research-frofa-fixture-20261030'
    checks['old_fixture_manifest_unchanged']=digest(old/'SHA256SUMS')==previous['fixture_manifest_sha256']
    checks['old_fixture_all_payload_hashes_unchanged']=all(digest(old/line.split('  ',1)[1])==line.split('  ',1)[0] for line in (old/'SHA256SUMS').read_text().splitlines())
    checks['git_head_unchanged']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()==previous['git_head']
    checks['git_index_unchanged']=digest(PROJECT/'.git/index')==previous['git_index_sha256']
    excludes={'SHA256SUMS','completion-audit.json'}
    paths=sorted(f for f in ROOT.rglob('*') if f.is_file() and f.name not in excludes)
    hashes={str(f.relative_to(ROOT)):digest(f) for f in paths}
    checks['cache_sha256']=hashes['cache/features.fp32.npy']==load('cache-lock.json')['sha256']
    checks['manifest_sha256']=hashes['manifest.json']==load('manifest-lock.json')['sha256']
    checks['model_sha256']=hashes['assets/siglip/model.safetensors']==p['assets']['model_sha256']
    checks['all_final_heads_sha256']=all(hashes[f['checkpoint']]==f['checkpoint_sha256'] for f in fits if f['phase']=='confirm')
    snapshot=load('artifact-hashes.json');snapshot_drift=[name for name,h in snapshot.items() if hashes.get(name)!=h]
    checks['worker_snapshot_only_expected_final_event_append']=snapshot_drift==['events.jsonl']
    with (ROOT/'SHA256SUMS').open('x') as out:
        for name,h in hashes.items():out.write(h+'  '+name+'\n')
    elapsed=time.time()-ledger['start_epoch'];disk=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    checks['final_archive_within_original_4h']=elapsed<14400
    checks['final_disk_within_8GiB']=disk<8589934592
    audit={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'all_passed':all(checks.values()),'checks':checks,'failed':[name for name,ok in checks.items() if not ok],'payload_files':len(paths),'payload_bytes':sum(f.stat().st_size for f in paths),'final_disk_bytes_before_audit':disk,'SHA256SUMS_sha256':digest(ROOT/'SHA256SUMS'),'archive_elapsed_from_first_download_seconds':elapsed,'worker_snapshot_drift_explained':snapshot_drift,'raw_ledger_note':'Use counts.complete_fits=36; top-level unused initialization field complete_fits=0 retained unchanged.','no_new_model_evaluation_or_training':True,'excluded_self_referential_files':sorted(excludes)}
    (ROOT/'completion-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
    if not audit['all_passed']:raise SystemExit(1)
if __name__=='__main__':main()
