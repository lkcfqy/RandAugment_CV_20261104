"""External 600-second watchdog; kills this fixture's process group on timeout."""
from portable_dev.paths import project_path as _portable_project_path
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PRESERVE=_portable_project_path(__file__, 'research-ctrla-mechanism-20261030')


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def snapshot():
    result={}
    for p in sorted(PRESERVE.rglob('*')):
        if p.is_file():
            s=p.stat()
            row={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
            if s.st_size<=2_000_000:
                row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            result[str(p.relative_to(PRESERVE))]=row
    return result


def main():
    assert not (ROOT/'protocol-lock.json').exists(), 'Do not overwrite a locked formal run'
    before=snapshot()
    write(ROOT/'ctrla-preservation-before.json',before)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('protocol.json','audit_fixture.py','run_bounded.py','source-receipts.json')}
    write(ROOT/'protocol-lock.json',{'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'sha256':hashes,'seed_retuning_allowed':False})
    env=dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    prior_wall=sum(json.loads(p.read_text())['wall_seconds'] for p in (ROOT/'attempts').glob('*/run-receipt.json'))
    remaining=600-prior_wall
    assert remaining>0, 'Aggregate formal compute budget exhausted'
    start=time.monotonic()
    timed_out=False
    command=[sys.executable,'-B',str(ROOT/'audit_fixture.py')]
    with (ROOT/'formal-run.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                code=proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                code=proc.wait(timeout=5)
    wall=time.monotonic()-start
    after=snapshot()
    write(ROOT/'ctrla-preservation-after.json',after)
    write(ROOT/'run-receipt.json',{'command':command,'wall_seconds':wall,'wall_limit_seconds':600,
                                  'timed_out':timed_out,'exit_code':code,
                                  'prior_formal_wall_seconds':prior_wall,'aggregate_formal_wall_seconds':prior_wall+wall,
                                  'ctrla_preservation_equal':before==after,
                                  'ctrla_files_checked':len(before),
                                  'preservation_scope':'Every file size/mtime; SHA256 for files <=2MB; no large weight reads',
                                  'environment_overrides':{k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}})
    print((ROOT/'run-receipt.json').read_text(),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
