"""Verify saved evidence only. No torch import, experiment execution or network."""
from portable_dev.paths import project_path as _portable_project_path
import ast
import datetime
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic()
    checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
    manifest=json.loads((ROOT/'artifact-manifest.json').read_text())
    for name,entry in manifest['files'].items():
        path=ROOT/name
        check('artifact/'+name,path.is_file() and digest(path)==entry['sha256'])
    receipts=json.loads((ROOT/'source-receipts.json').read_text())
    tree=json.loads((ROOT/'sources/tree.json').read_text())
    commit=json.loads((ROOT/'sources/commit.json').read_text())
    check('commit_sha',commit['sha']=='d854af15467285ad8f46500d6fc2da5225fb5bd0')
    check('tree_not_truncated',tree['truncated'] is False)
    by_path={e['path']:e for e in tree['tree']}
    for receipt in receipts:
        body=(ROOT/receipt['path']).read_bytes()
        check('source_sha256/'+receipt['path'],hashlib.sha256(body).hexdigest()==receipt['sha256'])
        if receipt.get('verified_against_pinned_tree'):
            blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
            check('git_blob/'+receipt['path'],blob==by_path[receipt['path'].removeprefix('sources/')]['sha']==receipt['git_blob_sha1'])
    directories={'':commit['tree']['sha']}
    directories.update({e['path']:e['sha'] for e in tree['tree'] if e['type']=='tree'})
    for parent,expected in directories.items():
        entries=[e for e in tree['tree'] if ('' if str(Path(e['path']).parent)=='.' else str(Path(e['path']).parent))==parent]
        entries.sort(key=lambda e:Path(e['path']).name+('/' if e['type']=='tree' else ''))
        body=b''.join(e['mode'].lstrip('0').encode()+b' '+Path(e['path']).name.encode()+b'\0'+bytes.fromhex(e['sha']) for e in entries)
        actual=hashlib.sha1(b'tree '+str(len(body)).encode()+b'\0'+body).hexdigest()
        check('git_tree/'+(parent or '/'),actual==expected)
    lock=json.loads((ROOT/'protocol-lock.json').read_text())
    for name,sha in lock['sha256'].items():
        check('locked/'+name,digest(ROOT/name)==sha)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    original=json.loads((ROOT/'attempts/attempt01-sandbox/protocol.json').read_text())
    check('no_protocol_retuning',protocol==original)
    result=json.loads((ROOT/'results/summary.json').read_text())
    run=json.loads((ROOT/'run-receipt.json').read_text())
    check('successful_fixture',result['checks']==105 and result['passed']==105 and result['failed']==0 and result['fatal_error'] is None)
    check('bounded_compute',run['exit_code']==0 and not run['timed_out'] and run['aggregate_formal_wall_seconds']<600)
    check('zero_training_compute',result['target_model_training_steps']==result['optimizer_steps']==result['colab_CU']==result['accelerator_tensor_operations_requested']==0)
    parsed=ast.parse((ROOT/'audit_fixture.py').read_text())
    functions=[ast.unparse(n.func) for n in ast.walk(parsed) if isinstance(n,ast.Call)]
    check('no_optimizer_step_or_accelerator_call',not any(n.endswith('.step') or '.cuda' in n or '.mps' in n for n in functions))
    check('no_dynamic_repo_execution',not any(n in ('exec','eval','compile','__import__') for n in functions))
    imported=[n.module for n in ast.walk(parsed) if isinstance(n,ast.ImportFrom)]
    imported += [a.name for n in ast.walk(parsed) if isinstance(n,ast.Import) for a in n.names]
    check('no_repository_import',not any(n and (n.split('.')[0] in ('train','Dataset','A2C','sources','augmentation','Network')) for n in imported))
    before=json.loads((ROOT/'ctrla-preservation-before.json').read_text())
    after=json.loads((ROOT/'ctrla-preservation-after.json').read_text())
    check('ctrla_saved_snapshots_equal',before==after)
    current={}
    old=_portable_project_path(__file__, 'research-ctrla-mechanism-20261030')
    for path in sorted(old.rglob('*')):
        if path.is_file():
            s=path.stat(); entry={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
            if s.st_size<=2_000_000:
                entry['sha256']=digest(path)
            current[str(path.relative_to(old))]=entry
    check('ctrla_current_matches_before',current==before)
    verification={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'wall_seconds':time.monotonic()-start,'checks':checks,
                  'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
                  'experiments_executed':0,'network_calls':0,
                  'ctrla_files':len(before),'ctrla_files_content_hashed':sum('sha256' in r for r in before.values())}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='checks'},indent=2))
    return int(any(not c['passed'] for c in checks))


if __name__=='__main__':
    raise SystemExit(main())
