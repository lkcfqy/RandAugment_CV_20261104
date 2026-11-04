"""Reused bounded stdout transport from previously exercised local Colab controller."""

import os,signal,subprocess,queue,threading,time

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
