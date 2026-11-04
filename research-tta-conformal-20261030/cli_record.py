import datetime, json, os, re, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
name,*args=sys.argv[1:]
started=time.time()
env=os.environ.copy();env['COLAB_CLI_ASSIGN_RETRIES']='0'
cmd=['/Users/lkc/.local/share/google-colab-cli/bin/python',str(ROOT/'colab_guard.py'),*args]
def clean(x):
    x=re.sub(r'ya29\.[A-Za-z0-9_.-]+','[REDACTED]',x)
    x=re.sub(r'(?i)(access_token|refresh_token|id_token|client_secret|authorization_code)([\s\"\x27:=]+)[^\s,\"\x27]+',r'\1\2[REDACTED]',x)
    x=re.sub(r'https://accounts\.google\.com/[^\s]+','[OAUTH URL OMITTED]',x)
    x=re.sub(r'(?i)(colab-runtime-proxy-token|runtime_proxy_token)([=\s\"\x27:]+)[^&\s,\"\x27]+',r'\1\2[REDACTED]',x)
    x=re.sub(r'(?i)Bearer\s+[A-Za-z0-9_.-]+','Bearer [REDACTED]',x)
    return x
try:
    r=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=180)
    code=r.returncode;out=clean(r.stdout);err=clean(r.stderr)
except subprocess.TimeoutExpired:
    code=124;out='';err='CLI_TIMEOUT: outcome uncertain; reconcile sessions before retry.'
rec={'started_epoch':started,'started_utc':datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),
     'finished_epoch':time.time(),'args':args,'exit_code':code,'stdout':out,'stderr':err,
     'new_oauth_enabled':False,'assignment_retries':0}
directory=ROOT/'cli-ledger';directory.mkdir(exist_ok=True)
p=directory/(name+'.json')
if p.exists():raise RuntimeError('Refusing to replace historical CLI receipt '+str(p))
p.write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec,indent=2),flush=True)
sys.exit(code)
