"""Reconstruct public data manifest only. No model, training, GPU, or inference."""
import ast,collections,hashlib,json,os,pickle,tarfile,time,urllib.request
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE/'recovered-data';ROOT.mkdir(exist_ok=True)
OUT=HERE/'partial-evidence';OUT.mkdir(exist_ok=True)
EVENTS=OUT/'cpu-recovery-events.jsonl'
ALLOCATED_AT=time.time();SOURCE_START=time.time()
source=(HERE/'remote_experiment.py').read_text()
tree=ast.parse(source)
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'event','save','sha','data_and_split'}]
assert len(selected)==4
ns=globals()
exec(compile(ast.Module(body=selected,type_ignores=[]),str(HERE/'remote_experiment.py'),'exec'),ns)
x,y,roles=data_and_split(np)
events=[json.loads(l) for l in (HERE/'monitor/remote-process-019.log').read_text().splitlines() if l.startswith('{')]
original=next(e for e in events if e.get('event')=='split_validated')
recovered=json.loads((OUT/'split-summary.json').read_text())
for role in roles:
    assert recovered['roles'][role]==original['roles'][role], 'Role reconstruction mismatch: '+role
assert recovered['archive_sha256']==original['archive_sha256']
assert recovered['duplicate_group_count']==original['duplicate_group_count']
np.save(OUT/'reconstructed-heldout-ids.npy',np.concatenate([roles[k] for k in ('pseudocal','validation','cal','eval')]))
report={'reconstruction_only':True,'original_runtime_artifact_retrieved':False,'no_training':True,
        'no_inference':True,'no_GPU':True,'official_archive_matches_original_SHA256':True,
        'all_role_ID_hashes_match_original_log':True,'source_code_sha256':sha(HERE/'remote_experiment.py'),
        'image_hash_manifest_reconstructed':True,'raw_public_archive_bytes':(ROOT/'cifar-100-python.tar.gz').stat().st_size,
        'elapsed_seconds':time.time()-SOURCE_START,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
save('cpu-provenance-recovery.json',report)
print(json.dumps(report,indent=2))
