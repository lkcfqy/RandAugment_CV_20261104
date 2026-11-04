import os,subprocess,json
from pathlib import Path
root=Path('/content/tta_conformal_20261030')
root.mkdir(exist_ok=True)
assert not (root/'started.json').exists(), 'Refusing duplicate launch'
env=os.environ.copy()
env.update(TTA_ALLOCATED_AT='1793382739.917252',TTA_VERIFIED_RATE='1.15',TTA_PROTOCOL_SHA256='28a8f92ca9fdd76cbcebe4998e9ff55f259f5e774a29dca190232f2eb5587f0b')
log=open(root/'process.log','w')
proc=subprocess.Popen(['python','-u','/content/tta_remote_experiment.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'started.json').write_text(json.dumps({'pid':proc.pid}))
print(json.dumps({'started':True,'pid':proc.pid,'protocol_sha256':'28a8f92ca9fdd76cbcebe4998e9ff55f259f5e774a29dca190232f2eb5587f0b'}))
