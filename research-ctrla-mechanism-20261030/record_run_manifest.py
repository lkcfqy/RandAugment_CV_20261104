from ctrla_adapter import *
from datetime import datetime,timezone
import subprocess,sys
assert (ROOT/'asset-manifest.json').exists()
assert not (ROOT/'run-manifest.json').exists()
protocol=json.loads((ROOT/'protocol-lock.json').read_text())
assert protocol['protocol_sha256']==sha256(ROOT/'protocol.json')
assert protocol['sampling_manifest_sha256']==sha256(ROOT/'sampling-manifest.json')
result={'created_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,'-B',str(ROOT/'run_stages.py')],'protocol_lock':protocol,'files':{p.name:sha256(p) for p in ROOT.glob('*') if p.suffix in ('.py','.md','.json','.txt') and p.name not in ('budget-state.json',)},'python_executable':sys.executable,'mps_fallback':'0','fast_math':'0','data_seed':20261030,'model_seeds':[0,1],'no_test_load':True,'colab_cu':0,'online_fork':False}
write_json(ROOT/'run-manifest.json',result)
(ROOT/'environment-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True))
budget=json.loads((ROOT/'budget-state.json').read_text());budget['synthetic_train_images']=2008;budget['synthetic_counter_correction']='125 first failed step + 1875 successful timed steps +8 gradient-comparison model forwards; zero real exposure';write_json(ROOT/'budget-state.json',budget)
print(json.dumps({'run_manifest_sha256':sha256(ROOT/'run-manifest.json'),'real_training_images_before':budget['train_images'],'inference_images_before':budget['inference_images']},indent=2))
