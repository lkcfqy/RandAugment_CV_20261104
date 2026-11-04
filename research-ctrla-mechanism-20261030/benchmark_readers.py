from ctrla_adapter import *
from frozen_diagnostics import ChallengeDataset
from train_common import loader
from datetime import datetime,timezone

def main():
    assert not (ROOT/'seed0-strength-match.json').exists()
    ds=ChallengeDataset('strength_match',[.7]*15,[.2]*15,1.,'F')
    small=torch.utils.data.Subset(ds,range(1000));results=[]
    for workers in (0,6):
        h=hashlib.sha256();t=time.monotonic();count=0
        for x,y in loader(small,workers=workers):
            h.update(x.numpy().tobytes());h.update(y.numpy().tobytes());count+=len(y)
        results.append({'workers':workers,'images':count,'elapsed_seconds':time.monotonic()-t,'tensor_and_label_sha256':h.hexdigest()})
    assert results[0]['tensor_and_label_sha256']==results[1]['tensor_and_label_sha256']
    write_json(ROOT/'reader-benchmark.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'model_forwards':0,'outcome_metrics_computed':False,'results':results,'byte_identical':True,'chosen_frozen_challenge_workers':0,'training_workers_unchanged':6,'challenge_manifest_unchanged':sha256(ROOT/'sampling-manifest.json'),'purpose':'Remove spawn/teardown overhead only; preserve exact CPU Pillow tensors and order.'})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
