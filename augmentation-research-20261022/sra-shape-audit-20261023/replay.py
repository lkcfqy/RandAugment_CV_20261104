"""Reproduce in a fresh sibling folder, preserving the original evidence."""
import os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
target=HERE.parent/('sra-shape-replay-'+stamp)
target.mkdir(exist_ok=False)
for name in ('protocol.json','protocol-lock.json','evidence-ledger.json','code-lock.json','mis_score.py','diagnose.py','verify_results.py'):
    shutil.copyfile(HERE/name,target/name)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
for name in ('diagnose.py','verify_results.py'):
    subprocess.run([sys.executable,'-B',str(target/name)],env=env,check=True,timeout=65)
print('Preserved replay:',target)
