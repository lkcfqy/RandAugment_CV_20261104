import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
a=json.loads((p/'cli-ledger/allocate-t4.json').read_text())
protocol=hashlib.sha256((p/'PROTOCOL_ZH.md').read_bytes()).hexdigest()
code="""import os,subprocess,json
from pathlib import Path
root=Path('/content/tta_conformal_20261030')
root.mkdir(exist_ok=True)
assert not (root/'started.json').exists(), 'Refusing duplicate launch'
env=os.environ.copy()
env.update(TTA_ALLOCATED_AT=%r,TTA_VERIFIED_RATE='1.15',TTA_PROTOCOL_SHA256=%r)
log=open(root/'process.log','w')
proc=subprocess.Popen(['python','-u','/content/tta_remote_experiment.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'started.json').write_text(json.dumps({'pid':proc.pid}))
print(json.dumps({'started':True,'pid':proc.pid,'protocol_sha256':%r}))
""" % (str(a['started_epoch']),protocol,protocol)
(p/'launch_remote.py').write_text(code)
(p/'check_remote.py').write_text("""import json
from pathlib import Path
r=Path('/content/tta_conformal_20261030')
p=r/'output/events.jsonl'
print('EVENTS:',p.read_text()[-14000:] if p.exists() else 'not yet created')
state=r/'output/run-state.json'
print('STATE:',state.read_text() if state.exists() else 'RUNNING')
log=r/'process.log'
print('LOG_TAIL:',log.read_text()[-3000:] if log.exists() else 'missing')
""")
print('launcher created, allocation start',a['started_utc'])
