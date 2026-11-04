from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
names=['PROTOCOL_ZH.md','SOURCES.md','conformal_core.py','math_acceptance.py','remote_experiment.py']
record={'frozen_utc':datetime.now(timezone.utc).isoformat(),'training_started':False,
        'seed':0,'split_seed':20261030,'gpu_limit_minutes':120,'CU_limit':6,
        'wall_limit_hours':6,'storage_limit_bytes':2*1024**3,
        'files':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names}}
Path('protocol-lock.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
