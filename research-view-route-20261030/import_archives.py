"""Import only byte-identical official Pets archives; never accept repacked datasets."""
import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
EXPECTED={'images':(791918971,'67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d','5c4f3ee8e5d25df40f4fd59a7f44e54c'),
          'annotations':(19173078,'52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91','95a8c909bbe2e81eed6a22bccdf3f68f')}
def digest(path,algorithm):
    with path.open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument('--images',type=Path,required=True);parser.add_argument('--annotations',type=Path,required=True)
parser.add_argument('--remove-task-created-archives',action='store_true',help='Only files inside this task assets directory; never historical external caches')
args=parser.parse_args();assert time.time()<datetime(2026,10,31,1,14,tzinfo=timezone.utc).timestamp(),'Original wall deadline expired'
assert not (ROOT/'protocol.lock.json').exists(),'No asset replacement after protocol lock'
receipts=[]
for name in ['annotations','images']:
    p=getattr(args,name).resolve();size,sha,md5=EXPECTED[name]
    assert p.stat().st_size==size and digest(p,'sha256')==sha and digest(p,'md5')==md5,(name,'Official archive checksum mismatch')
    if args.remove_task_created_archives:assert p.parent==(ROOT/'assets').resolve(),'Refuse to delete outside task-created assets'
    receipts.append({'asset':name,'input_path':str(p),'bytes':size,'sha256':sha,'md5':md5,'verified_before_extract':True})
for name in ['annotations','images']:
    p=getattr(args,name).resolve()
    with tarfile.open(p,'r:gz') as tar:
        selected=[]
        for member in tar.getmembers():
            part=Path(member.name)
            assert not part.is_absolute() and '..' not in part.parts,'Unsafe tar path'
            if not member.isfile():continue
            if name=='images' and part.suffix.lower()=='.jpg' and part.parts[0]=='images':selected.append(member)
            if name=='annotations' and str(part) in ['annotations/trainval.txt','annotations/test.txt','annotations/list.txt','annotations/README']:selected.append(member)
        occupied=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())
        assert occupied+sum(x.size for x in selected)<=3*1024**3,'Extraction exceeds 3GiB'
        for member in selected:
            target=ROOT/'data'/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists(),'Refuse to overwrite an existing extracted file'
            with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    if args.remove_task_created_archives:p.unlink()
result={'status':'complete','mode':'verified-byte-identical-archive-import','utc':datetime.now(timezone.utc).isoformat(),
        'assets':receipts,'task_created_archives_deleted':args.remove_task_created_archives,'historical_cache_deleted':False,
        'checksum_source':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/oxford_iiit_pet/checksums.tsv',
        'md5_source':'installed official torchvision 0.28.0 OxfordIIITPet._RESOURCES'}
(ROOT/'asset-preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
