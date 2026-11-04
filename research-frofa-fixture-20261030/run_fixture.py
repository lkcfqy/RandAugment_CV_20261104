"""Persistent continuous wall-clock budget; no reset across attempts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BUDGET = ROOT/'budget.json'


def utc():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    seal = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['protocol.json', 'DERIVATION.md', 'controls.py', 'fixture.py', 'run_fixture.py']}
    if BUDGET.exists():
        state = json.loads(BUDGET.read_text())
        if state.get('completed'):
            raise SystemExit('Formal computation already completed; refusing extra run.')
        if state['sealed_sha256'] != seal:
            raise SystemExit('Sealed artifacts differ; refusing unregistered retry.')
    else:
        now = time.time()
        state = dict(start_epoch=now, deadline_epoch=now+protocol['budget_seconds'],
                     start_utc=utc(), start_monotonic=time.monotonic(),
                     budget_seconds=protocol['budget_seconds'], sealed_sha256=seal,
                     attempts=[], completed=False)
        write_json(BUDGET, state)
    remaining = min(state['deadline_epoch']-time.time(),
                    state['budget_seconds']-(time.monotonic()-state['start_monotonic']))
    if remaining <= 0:
        raise SystemExit('Continuous 600-second deadline exhausted; stopping.')
    index = len(state['attempts'])+1
    attempt = dict(index=index, started_utc=utc(), remaining_seconds=remaining,
                   command=[sys.executable, '-B', str(ROOT/'fixture.py')])
    state['attempts'].append(attempt)
    write_json(BUDGET, state)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', FIXTURE_ROOT=str(ROOT))
    log = ROOT/f'worker-attempt-{index:02d}.log'
    with log.open('xb') as out:
        proc = subprocess.Popen(attempt['command'], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        attempt['pid'] = proc.pid
        write_json(BUDGET, state)
        try:
            code = proc.wait(timeout=max(0.001, remaining-0.1))
            attempt['exit_code'] = code
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            attempt['exit_code'] = proc.returncode
            attempt['timeout_killed'] = True
    attempt['finished_utc'] = utc()
    state['elapsed_continuous_seconds'] = time.monotonic()-state['start_monotonic']
    state['elapsed_epoch_seconds'] = time.time()-state['start_epoch']
    state['finished_utc'] = utc()
    state['completed'] = attempt['exit_code'] == 0
    state['within_budget'] = state['elapsed_continuous_seconds'] <= state['budget_seconds']
    write_json(BUDGET, state)
    print(json.dumps(state, indent=2))
    raise SystemExit(attempt['exit_code'])


if __name__ == '__main__':
    main()
