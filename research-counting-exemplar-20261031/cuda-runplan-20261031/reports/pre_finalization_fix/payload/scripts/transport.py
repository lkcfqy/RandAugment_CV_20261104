"""Small streaming/persistence helpers. No service calls on import."""
import hashlib,os,queue,signal,subprocess,threading,time
from pathlib import Path
from contracts import PREFIX,decode_packet,validate_artifact_path
from evidence import atomic_bytes

class Collector:
    def __init__(self,root,max_stdout_bytes=256*1024**2):
        self.root=Path(root);self.artifacts=self.root/'artifacts';self.root.mkdir(parents=True,exist_ok=True);self.max_stdout_bytes=max_stdout_bytes
    def line(self,line):
        log=self.root/'stdout.log'
        if (log.stat().st_size if log.exists() else 0)+len(line.encode())>self.max_stdout_bytes:raise RuntimeError('local evidence transfer/disk reserve reached')
        with log.open('ab') as stream:
            stream.write(line.encode());stream.flush();os.fsync(stream.fileno())
        if not line.startswith(PREFIX):return
        path,data=decode_packet(line.rstrip('\r\n'));destination=self.artifacts/path
        mutable=path.startswith(('engineering/','attempts/')) or path in ('artifact_index.json','supervisor.json')
        if destination.exists() and destination.read_bytes()!=data and not mutable:
            raise ValueError('conflicting immutable artifact: '+path)
        atomic_bytes(destination,data)
    def missing(self,index):
        missing=[]
        for rel,item in index['files'].items():
            path=self.artifacts/validate_artifact_path(rel)
            if not path.is_file():missing.append(rel);continue
            data=path.read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:missing.append(rel)
        return missing

def terminate_group(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=2)

def bounded_stream(command,on_line,should_stop,*,cwd=None,env=None,poll_seconds=.1,on_started=None):
    """External watchdog; reader thread cannot block deadline checks on readline."""
    process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    events=queue.Queue();start=time.monotonic();stopped=False
    def reader():
        try:
            for line in process.stdout:events.put(line)
        finally:events.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start();eof=False
    try:
        if on_started:on_started(process.pid)
        while not eof:
            if should_stop():stopped=True;terminate_group(process)
            try:line=events.get(timeout=poll_seconds)
            except queue.Empty:continue
            if line is None:eof=True
            else:on_line(line)
        code=process.wait(timeout=2)
        return {'returncode':code,'terminated_by_watchdog':stopped,'wall_seconds':time.monotonic()-start,'pid':process.pid}
    finally:
        terminate_group(process);process.stdout.close();thread.join(timeout=1)
