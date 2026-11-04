"""Every manufactured oracle, under a cumulative 180-second worker wall cap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,resource,subprocess,sys,time,unittest
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LEDGER=ROOT/'reports/oracle_runs.json'
WORKER_RESULT=ROOT/'reports/oracle_worker_result.json'

def worker():
    cpu=time.process_time();t=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests':result.testsRun,'failures':[str(x[0]) for x in result.failures],'errors':[str(x[0]) for x in result.errors],'test_wall_seconds':time.monotonic()-t,'cpu_seconds':time.process_time()-cpu,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'torch_imported':'torch' in sys.modules}
    WORKER_RESULT.write_text(json.dumps(report,indent=2)+'\n')
    return int(not result.wasSuccessful())

def main():
    ROOT.joinpath('reports').mkdir(exist_ok=True)
    runs=json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    spent=sum(x['wall_seconds'] for x in runs)
    if spent>=180:raise SystemExit('CPU oracle wall budget exhausted')
    WORKER_RESULT.unlink(missing_ok=True);t=time.monotonic()
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    timed_out=False
    try:out,err=child.communicate(timeout=180-spent)
    except subprocess.TimeoutExpired:
        child.kill();out,err=child.communicate();timed_out=True
    except BaseException:
        child.kill();child.communicate();raise
    report=json.loads(WORKER_RESULT.read_text()) if WORKER_RESULT.exists() else {'tests':0,'failures':[],'errors':['worker did not complete'],'cpu_seconds':None}
    ROOT.joinpath('reports/oracle_last_output.txt').write_text(out+err)
    report.update(utc=datetime.now(timezone.utc).isoformat(),kind='full_oracle_suite',wall_seconds=time.monotonic()-t,exit_code=child.returncode,timed_out=timed_out,synthetic_only=True,real_images_opened=0,real_model_forwards=0,gpu_seconds=0)
    runs.append(report);LEDGER.write_text(json.dumps(runs,indent=2)+'\n')
    print(out+err,end='');print('ORACLE_CUMULATIVE_WALL_SECONDS',sum(x['wall_seconds'] for x in runs))
    return 1 if timed_out else child.returncode

if __name__=='__main__':sys.exit(worker() if '--worker' in sys.argv else main())
