from ctrla_adapter import *
from train_common import TrainDataset,loader,fit_statistics
from datetime import datetime,timezone
import shutil,pickle

def main():
    b=json.loads((ROOT/'budget-state.json').read_text());assert b['train_images']==0
    for name in ('common-stage-status.json','supervisor-result.json'):
        source=ROOT/name;destination=ROOT/('failed-spawn-'+name)
        if source.exists() and not destination.exists():shutil.copyfile(source,destination)
    p=ROOT/'weights'/'seed0-epoch00.pt';saved=ROOT/'weights'/'failed-spawn-seed0-epoch00.pt'
    if p.exists() and not saved.exists():shutil.copyfile(p,saved)
    stats=fit_statistics();ds=TrainDataset([0.]*15,[0.]*15,stats['mean'],stats['std'])
    pickle.loads(pickle.dumps(ds))
    it=iter(loader(ds,seed=1,workers=2));x,y=next(it);assert x.shape==(125,3,32,32) and y.shape==(125,)
    del it
    write_json(ROOT/'recovery-spawn.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'failure':'Dynamic official module omitted from sys.modules, so spawn could not pickle ControlAugment class','change':'Register audited module under its named key before exec_module, reproduced when child imports adapter','worker_smoke':'2 spawn workers, one125-image CPU preprocessing batch passed; no model forward or real training step','real_train_images_before':b['train_images'],'inference_images_before':b['inference_images'],'same_wall_deadline':b['deadline_unix'],'protocol_unchanged':sha256(ROOT/'protocol.json'),'updated_code':{n:sha256(ROOT/n) for n in ('ctrla_adapter.py','run_stages.py')},'scientific_parameters_changed':False,'authorized_recovery':True})
    print('spawn smoke passed; original deadline and data preserved',flush=True)

if __name__=='__main__':main()
