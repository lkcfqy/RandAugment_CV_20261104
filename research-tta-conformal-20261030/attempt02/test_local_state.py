"""CPU-only serialization control. No model updates or inference."""
import json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from state_io import *
R=Path(__file__).resolve().parent
O=R/'local-acceptance';O.mkdir(exist_ok=True)
npstate=np.random.get_state()
state={'phase':'fixture_no_training','model':{'w':torch.arange(24,dtype=torch.float32).reshape(4,6),
       'bn_running_mean':torch.zeros(4),'bn_counter':torch.tensor(12)},
       'optimizer':{'state':{0:{'momentum_buffer':torch.linspace(-1,1,24).reshape(4,6)}},'param_groups':[{'lr':.1,'params':[0]}]},
       'scheduler':{'implementation':'manual_cosine','base_lr':.1,'Tmax':25040,'next_step':12},
       'rng':{'python':random.getstate(),'numpy':[npstate[0],npstate[1].tolist(),int(npstate[2]),int(npstate[3]),float(npstate[4])],
              'torch_cpu':torch.get_rng_state(),'torch_cuda':[]},
       'permutation':torch.arange(40000),'step':12,'epoch':0,'cursor':12,'presentations':1536,
       'normalization':{'mean':torch.tensor([.5,.5,.5]),'std':torch.tensor([.2,.2,.2])}}
identity={'run_id':'local_serialization_control','protocol':'fixture','no_training':True}
p=O/'source.pt';receipt=save_checkpoint(p,state,identity)
dest=O/'persisted.pt';shutil.copyfile(p,dest)
with dest.open('rb') as f:os.fsync(f.fileno())
data=validate_checkpoint(dest,receipt,identity)
assert canonical_hash(state)==canonical_hash(data['state'])
bad=O/'corrupt.pt';shutil.copyfile(dest,bad)
with bad.open('r+b') as f:f.seek(-16,2);f.write(b'INVALID-BYTES!!!!')
rejected=False
try:validate_checkpoint(bad,receipt,identity)
except AssertionError:rejected=True
assert rejected
identity_rejected=False
try:validate_checkpoint(dest,receipt,{'run_id':'wrong'})
except AssertionError:identity_rejected=True
assert identity_rejected
result={'passed':True,'torch':str(torch.__version__),'no_GPU':True,'training_updates':0,'inference_views':0,
        'state_roundtrip_exact':True,'model_optimizer_scheduler_RNG_cursor_present':True,
        'corruption_rejected':rejected,'identity_mismatch_rejected':identity_rejected,
        'file_sha256':receipt['sha256'],'content_sha256':receipt['state_content_sha256'],'bytes':receipt['bytes'],
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
atomic_json(O/'acceptance.json',result);print(json.dumps(result,indent=2))
