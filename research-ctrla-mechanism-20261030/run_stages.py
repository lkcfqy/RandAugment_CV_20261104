"""Hard-deadline local supervisor. No Colab calls and no online branches."""
from pathlib import Path
import subprocess,time,signal,os,json,sys,traceback
from ctrla_adapter import ROOT,write_json

def main():
    started=time.time();stages=[]
    try:
        for filename in ('train_common.py','frozen_diagnostics.py'):
            state=json.loads((ROOT/'budget-state.json').read_text())
            remaining=state['deadline_unix']-time.time()-10
            if remaining<=0:raise RuntimeError('No wall budget remains')
            logfile=ROOT/'logs'/f'{Path(filename).stem}.stdout.txt'
            attempt=1
            while logfile.exists():
                attempt+=1;logfile=ROOT/'logs'/f'{Path(filename).stem}-attempt{attempt:02d}.stdout.txt'
            with logfile.open('x') as out:
                p=subprocess.Popen([sys.executable,'-B',str(ROOT/filename)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(ROOT/'supervisor-live.json',{'stage':filename,'pid':p.pid,'deadline_unix':state['deadline_unix'],'started_unix':time.time(),'log':str(logfile.relative_to(ROOT))})
                try:code=p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise RuntimeError('HARD_8H_DEADLINE: stage stopped; preserve latest checkpoint')
            stages.append({'stage':filename,'exit_code':code})
            if code:raise RuntimeError(f'{filename} exited {code}; subsequent stage not started')
        write_json(ROOT/'supervisor-result.json',{'status':'complete','stages':stages,'elapsed_seconds':time.time()-started,'online_branch_started':False,'colab_cu':0})
    except Exception as e:
        write_json(ROOT/'supervisor-result.json',{'status':'stopped','stages':stages,'error':str(e),'traceback':traceback.format_exc(),'online_branch_started':False});raise

if __name__=='__main__':main()
