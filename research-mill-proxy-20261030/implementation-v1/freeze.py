"""Metadata/source freeze only: no real image decode and no model load/forward."""
from portable_dev.paths import project_path as _portable_project_path
import ast
import importlib.metadata
import json
import math
from datetime import datetime,timezone
from pathlib import Path
from millproxy.audit import code_manifest,current_code_hash,file_sha,expected_forward_keys
from millproxy.model import SOURCE_SHA256,WEIGHT_SHA,SOURCE_COMMIT
from millproxy.runtime import ComputeLedger,DiskGuard

ROOT=Path(__file__).resolve().parent
ASSETS=_portable_project_path(__file__, 'research-mill-proxy-20261030')
now=datetime.now(timezone.utc)
protocol=json.loads((ROOT/'protocol.json').read_text())
runs=json.loads((ROOT/'reports/oracle_runs.json').read_text())
last=next(r for r in reversed(runs) if r['tests']>0)
if last['failures'] or last['errors'] or last.get('timed_out'):raise SystemExit('Latest oracle suite is not green')
if last.get('torch_imported',False):raise SystemExit('Oracle unexpectedly imported torch')
spent=sum(r['wall_seconds'] for r in runs);charge=sum(math.ceil(r['wall_seconds']) for r in runs)
if spent>180:raise SystemExit('Oracle cap exceeded')
elapsed=(now-datetime.fromisoformat(protocol['implementation_start_utc'].replace('Z','+00:00'))).total_seconds()
if elapsed>3600:raise SystemExit('Implementation wall cap exceeded')
manifest=code_manifest();freeze_sha=current_code_hash()
for relative in manifest:
    p=ROOT/relative
    if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
source_checks={}
for rel,expected in SOURCE_SHA256.items():
    actual=file_sha(ASSETS/'vendor/HVI-CIDNet'/rel)
    if actual!=expected:raise SystemExit('Pinned source differs: '+rel)
    source_checks[rel]={'expected':expected,'actual':actual,'match':True}
if file_sha(ASSETS/'models/model.safetensors')!=WEIGHT_SHA:raise SystemExit('Checkpoint hash differs')
if file_sha(ASSETS/'reports/panel.json')!=protocol['panel_sha256']:raise SystemExit('Panel changed')
test_ast=ast.parse((ROOT/'tests/test_oracles.py').read_text())
tests=[{'name':n.name,'line':n.lineno} for n in ast.walk(test_ast) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
disk_guard=DiskGuard(ASSETS);disk_guard.reserve(1024*1024)
# A no-active-session ledger is a metadata record, not a computation release.
ledger=ComputeLedger(ROOT/'run-state/compute_ledger.json',oracle_charge=charge)
report={
 'status':'FROZEN_IMPLEMENTATION_ONLY_PENDING_FINAL_INDEPENDENT_REVIEW',
 'created_utc':now.isoformat(),'freeze_sha256':freeze_sha,'code_files':manifest,
 'official_source_commit':SOURCE_COMMIT,'official_source_hash_checks':source_checks,
 'model_safetensors_sha256':WEIGHT_SHA,'panel_sha256':protocol['panel_sha256'],
 'python_packages':{name:importlib.metadata.version(name) for name in ('numpy','Pillow','torch','safetensors','einops','huggingface_hub')},
 'last_oracle_tests':last['tests'],'last_oracle_failures':last['failures'],'last_oracle_errors':last['errors'],
 'last_oracle_output_sha256':file_sha(ROOT/'reports/oracle_last_output.txt'),'oracle_runs_sha256':file_sha(ROOT/'reports/oracle_runs.json'),
 'oracle_cumulative_measured_wall_seconds':spent,'conservative_oracle_charge_seconds':charge,
 'compute_ledger':'run-state/compute_ledger.json','remaining_original_compute_seconds':ledger.remaining,
 'implementation_elapsed_wall_seconds':elapsed,'disk_allocated_bytes_at_freeze':disk_guard.used(),
 'formal_expected_forward_identities':len(expected_forward_keys()),'real_images_read_for_research':0,'real_fit':0,'real_synthesis':0,'real_task_forwards':0,'real_utility_metrics':0,'GPU_seconds':0,'Colab_allocations':0,
 'manufactured_pipeline_note':'NumPy input-only callable over 326 manufactured identities; no CIDNet/torch import.',
 'fixed_degeneracy':{'synthetic_degeneracy':protocol['synthetic_degeneracy'],'unidentifiable_policy':protocol['unidentifiable_policy']},
 'pending':['Final independent static review on this freeze','Later authorized full-resolution no-utility preflight and resource bound; no runtime feasibility claim yet'],
 'permission_environment_blockers':[],'denied_actions':[],'uncertain_actions':[]
}
(ROOT/'reports/freeze.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
(ROOT/'reports/TEST_INVENTORY.json').write_text(json.dumps(tests,indent=2)+'\n')
(ROOT/'reports/CODE_SHA256SUMS').write_text('\n'.join(f'{digest}  {path}' for path,digest in manifest.items())+'\n')
(ROOT/'STATUS.json').write_text(json.dumps({k:report[k] for k in ('status','created_utc','freeze_sha256','last_oracle_tests','oracle_cumulative_measured_wall_seconds','conservative_oracle_charge_seconds','real_fit','real_synthesis','real_task_forwards','GPU_seconds','permission_environment_blockers','pending')},indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','freeze_sha256','last_oracle_tests','oracle_cumulative_measured_wall_seconds','conservative_oracle_charge_seconds','implementation_elapsed_wall_seconds','disk_allocated_bytes_at_freeze','real_task_forwards','GPU_seconds','pending')},indent=2))
