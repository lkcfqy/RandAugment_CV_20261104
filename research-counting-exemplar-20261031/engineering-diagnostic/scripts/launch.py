"""Externally supervise one authorized run. Refuses any second launch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from controls import claim_launch, supervise

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parent

if __name__ == '__main__':
    worker = PHASE/'scripts/worker.py'
    python = ROOT/'cpu-readiness/.venv/bin/python'
    claim_launch(PHASE/'launch_once.json', {'supervisor_pid':os.getpid(),'created_unix':time.time(),
                 'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
                 'scope':'separately authorized diagnostic: CPU F then MPS F only; no dataset, no retries',
                 'wall_limit_seconds':120,'max_forward_attempts':2,'rss_limit_bytes':8*1024**3,'mps_limit_bytes':8*1024**3})
    result = supervise([str(python),'-B',str(worker)],PHASE/'run',120,8*1024**3,8*1024**3)
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['returncode'] == 0 and result['termination_reason'] is None else 2)
