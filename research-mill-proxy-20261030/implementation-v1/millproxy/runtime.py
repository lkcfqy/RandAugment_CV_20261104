"""Continuous compute budget and bounded preflight; never run on import."""
import subprocess
import time
import json
import os
import math
from pathlib import Path
from .audit import require_release,resource_bound

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

class DiskGuard:
    def __init__(self,root,limit_bytes=2*1024**3):self.root=Path(root);self.limit=int(limit_bytes)
    def used(self):return sum(p.stat().st_blocks*512 for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
    def reserve(self,additional_bytes=0):
        if additional_bytes<0:raise ValueError('Negative disk reservation')
        if self.used()+int(additional_bytes)>self.limit:raise TimeoutError('RESOURCE_BLOCKED: 2 GiB directory disk cap')

class ComputeLedger:
    """Persistent charge account. An unfinished session blocks all new sessions.
    Lock files are never silently deleted or treated as a budget reset.
    """
    def __init__(self,path,*,oracle_charge,clock=time.monotonic):
        self.path=Path(path);self.lock=self.path.with_suffix('.lock');self.clock=clock
        if oracle_charge<0:raise ValueError('Negative oracle charge')
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'limit_seconds':1800.,'oracle_charge_seconds':0.,'events':[],'active':None}
        if self.state['limit_seconds']!=1800:raise ValueError('Original compute cap changed')
        if oracle_charge<self.state['oracle_charge_seconds']:raise ValueError('Oracle charge cannot decrease')
        if oracle_charge!=self.state['oracle_charge_seconds']:
            if self.state['active'] is not None:raise ValueError('Cannot change oracle ledger during an active session')
            self.state['oracle_charge_seconds']=float(oracle_charge)
        atomic_json(self.path,self.state)
    @property
    def spent(self):return self.state['oracle_charge_seconds']+sum(e['charged_seconds'] for e in self.state['events'])
    @property
    def remaining(self):return max(0.,1800.-self.spent)
    def begin(self,kind):
        if kind not in ('preflight','formal'):raise ValueError('Unknown compute stage')
        if self.state['active'] is not None or self.lock.exists():raise ValueError('Prior session pending/uncertain; inspect it, never reset budget')
        if self.remaining<=0:raise TimeoutError('RESOURCE_BLOCKED: total 30-minute compute budget exhausted')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:fd=os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as e:raise ValueError('Another compute session is active') from e
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        self.state['active']={'kind':kind,'started_monotonic':self.clock(),'reserved_seconds':min(120.,self.remaining) if kind=='preflight' else self.remaining,'spent_before':self.spent}
        atomic_json(self.path,self.state)
        return self.state['active'].copy()
    def finish(self,status):
        active=self.state['active']
        if active is None:raise ValueError('No active session')
        duration=self.clock()-active['started_monotonic']
        if duration<0:raise ValueError('Clock epoch changed; session remains uncertain')
        # Round upward to cover final ledger flush/lock removal overhead. Never
        # reduce a measured charge or create extra time by restarting a stage.
        event={**active,'wall_seconds':duration,'charged_seconds':float(math.ceil(duration)),'status':status}
        self.state['events'].append(event);self.state['active']=None
        atomic_json(self.path,self.state);self.lock.unlink()
        return event

class ContinuousBudget:
    def __init__(self,remaining_seconds,*,prior_spent=0,disk_root=None):
        if not 0<remaining_seconds<=1800 or prior_spent<0 or remaining_seconds+prior_spent>1800:raise ValueError('Budget cannot exceed original 30 minutes')
        self.start=time.monotonic();self.limit=float(remaining_seconds);self.prior=float(prior_spent)
        self.disk=DiskGuard(disk_root) if disk_root is not None else None
    def check(self):
        if self.elapsed>=self.limit:raise TimeoutError('RESOURCE_BLOCKED: continuous computation deadline reached')
        if self.disk is not None:self.disk.reserve()
    def before_write(self,bytes_needed):
        self.check()
        if self.disk is not None:self.disk.reserve(int(bytes_needed)+65536)
    @property
    def elapsed(self):return time.monotonic()-self.start
    @property
    def total_spent(self):return self.prior+self.elapsed
    @property
    def remaining(self):return max(0.,self.limit-self.elapsed)

def preflight(forward_probe,full_cell_probe,synchronize,*,t_spent,t_post,receipt,freeze_sha):
    """Probe supplied by a reviewed caller: 1 warmup + 3 batch-one forwards,
    one fixed full-resolution A/B x four-draw generation+stat cell. No utility
    metrics. t_post is an explicit conservative allowance for uncounted work,
    including fitting, final aggregation and persistence; never guessed here.
    """
    require_release(receipt,freeze_sha)
    start=time.monotonic();budget=ContinuousBudget(min(120.,1800.-t_spent),prior_spent=t_spent)
    forward_probe();synchronize();budget.check()
    timings=[]
    for _ in range(3):
        synchronize();t=time.monotonic();forward_probe();synchronize();timings.append(time.monotonic()-t);budget.check()
    t=time.monotonic();full_cell_probe();synchronize();cell=time.monotonic()-t;budget.check()
    elapsed=time.monotonic()-start;bound=resource_bound(t_spent+elapsed,max(timings),cell,t_post)
    return {'warmup_forwards':1,'timed_forwards':3,'three_forward_seconds':timings,'t_forward':max(timings),'t_cell':cell,'t_post':t_post,'preflight_seconds':elapsed,'t_spent':t_spent+elapsed,'T_bound':bound,'resource_go':bound<=1800,'utility_metrics_computed':False}

def run_worker_bounded(command,*,remaining_seconds,cwd):
    """External process watchdog also covers a stuck native numerical call.
    Caller must record elapsed time in the persistent 1800-second ledger.
    No shell or credentials; no retries or budget reset on timeout.
    """
    if not 0<remaining_seconds<=1800:raise ValueError('Invalid remaining budget')
    t=time.monotonic();p=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        out,err=p.communicate(timeout=remaining_seconds)
        return {'exit_code':p.returncode,'stdout':out,'stderr':err,'wall_seconds':time.monotonic()-t,'resource_blocked':False}
    except subprocess.TimeoutExpired:
        p.kill();out,err=p.communicate()
        return {'exit_code':p.returncode,'stdout':out,'stderr':err,'wall_seconds':time.monotonic()-t,'resource_blocked':True,'scientific_negative':False}
    except BaseException:
        p.kill();p.communicate()
        raise
