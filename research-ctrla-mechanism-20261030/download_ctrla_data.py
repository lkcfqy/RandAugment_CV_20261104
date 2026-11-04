from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib,json,tarfile,time
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url,check_integrity
ROOT=_portable_project_path(__file__, 'research-ctrla-mechanism-20261030')
data=ROOT/'data'; started=time.monotonic()
class TrainOnlyCIFAR10(CIFAR10):
    test_list=[]
    def __init__(self,root): super().__init__(root,train=True,download=False)
download_url(CIFAR10.url,str(data),filename=CIFAR10.filename,md5=CIFAR10.tgz_md5)
archive=data/CIFAR10.filename
assert check_integrity(archive,CIFAR10.tgz_md5)
allowed={f'{CIFAR10.base_folder}/{n}' for n,_ in CIFAR10.train_list}|{f'{CIFAR10.base_folder}/{CIFAR10.meta["filename"]}'}
with tarfile.open(archive) as tar:
    selected=[m for m in tar.getmembers() if m.name in allowed]
    assert len(selected)==6 and all(m.isfile() for m in selected)
    tar.extractall(data,members=selected,filter='data')
assert not (data/CIFAR10.base_folder/'test_batch').exists()
dataset=TrainOnlyCIFAR10(data)
labels=np.asarray(dataset.targets,dtype=np.int64)
np.save(data/'train_images.npy',dataset.data,allow_pickle=False)
np.save(data/'train_labels.npy',labels,allow_pickle=False)
rng=np.random.default_rng(20261030)
splits={k:[] for k in ('fit','controller','strength_match','audit')}
for cls in range(10):
    idx=rng.permutation(np.flatnonzero(labels==cls))
    splits['fit'].extend(idx[:4600]);splits['controller'].extend(idx[4600:4700]);splits['strength_match'].extend(idx[4700:4800]);splits['audit'].extend(idx[4800:])
for key,idx in splits.items():
    # Independent random ordering avoids deterministic calibration sign/class confounding.
    np.save(data/f'{key}_indices.npy',rng.permutation(idx),allow_pickle=False)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest={'official_url':CIFAR10.url,'torchvision_expected_archive_md5':CIFAR10.tgz_md5,'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,'train_member_md5':dict(CIFAR10.train_list),'files':{str(p.relative_to(ROOT)):digest(p) for p in data.rglob('*') if p.is_file() and p!=archive},'split_seed':20261030,'split_counts':{k:len(v) for k,v in splits.items()},'stratification_per_class':{'fit':4600,'controller':100,'strength_match':100,'audit':200},'test_batch_extracted':False,'test_dataset_instantiated':False,'archive_contains_unopened_test_member':True,'prior_use':'No traceable CIFAR-10 checkpoint or local dataset identified; broader prior research usage cannot be excluded. This is a new diagnostic partition, not a claim of independent confirmation.','elapsed_seconds':time.monotonic()-started}
(ROOT/'asset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2),flush=True)
