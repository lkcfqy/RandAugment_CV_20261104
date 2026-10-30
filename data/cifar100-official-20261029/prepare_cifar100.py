#!/usr/bin/env python3
"""Official binary CIFAR-100, train-only extraction and a fixed development split."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import numpy as np

ROOT = Path('/Users/lkc/Downloads/projects/RandAugment_CV_20261104/data/cifar100-official-20261029')
URL = 'https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz'
MD5 = '03b5dce01913d631647c71ecec9e9cb8'
OFFICIAL_PAGE = 'https://cave.cs.toronto.edu/kriz/cifar.html'
SEED = 'rand-cifar100-train-development-v1-20261029'


def digest_file(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    start=time.perf_counter();times={}
    ROOT.mkdir(parents=True,exist_ok=False)
    (ROOT/'.gitignore').write_text('*\n!.gitignore\n!README.md\n!preparation.json\n!prepare_cifar100.py\n')
    report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_url':URL,'official_page':OFFICIAL_PAGE,'official_archive_md5':MD5,'test_policy':'Archive retained opaque; test member not extracted, parsed, evaluated or used for splitting. Tar headers inspected for safety.','training':False}
    try:
        t=time.perf_counter()
        with urllib.request.urlopen(OFFICIAL_PAGE,timeout=45) as response:
            html=response.read(2*1024*1024)
        assert MD5.encode() in html and b'cifar-100-binary.tar.gz' in html
        (ROOT/'official-download-page.html').write_bytes(html)
        report['official_page_sha256']=hashlib.sha256(html).hexdigest()
        times['official_page_seconds']=time.perf_counter()-t
        t=time.perf_counter();size=0
        partial=ROOT/'cifar-100-binary.tar.gz.partial'
        with urllib.request.urlopen(URL,timeout=45) as response,partial.open('xb') as out:
            assert urllib.parse.urlparse(response.url).hostname in {'cave.cs.toronto.edu','www.cs.toronto.edu','cs.toronto.edu'}
            report['final_download_url']=response.url
            for block in iter(lambda:response.read(1024*1024),b''):
                size+=len(block)
                if size>250*1024*1024:raise ValueError('Archive exceeded 250 MiB cap')
                out.write(block)
                if size%(16*1024*1024)==0:print(json.dumps({'downloaded_MiB':size//1024**2}),flush=True)
        times['download_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        assert digest_file(partial,'md5')==MD5,'official MD5 mismatch; no extraction'
        report['archive_sha256']=digest_file(partial);report['archive_bytes']=size
        archive=ROOT/'cifar-100-binary.tar.gz';partial.rename(archive)
        times['archive_hash_seconds']=time.perf_counter()-t
        t=time.perf_counter();members=[]
        allowed={'cifar-100-binary/train.bin','cifar-100-binary/fine_label_names.txt','cifar-100-binary/coarse_label_names.txt'}
        known=allowed|{'cifar-100-binary/test.bin','cifar-100-binary'}
        extracted=ROOT/'train-only';extracted.mkdir()
        with tarfile.open(archive,'r:gz') as tar:
            items=tar.getmembers()
            for member in items:
                name=member.name.rstrip('/');p=PurePosixPath(name)
                assert not p.is_absolute() and '..' not in p.parts and name in known,name
                assert member.isdir() or member.isfile(),'link/device/other tar member rejected'
                assert member.size<=160*1024*1024,'member size limit'
                members.append({'name':name,'size':member.size,'extracted':name in allowed})
            assert len({m.name for m in items})==len(items),'duplicate member names'
            for member in items:
                if member.name not in allowed:continue
                destination=extracted/PurePosixPath(member.name).name
                with tar.extractfile(member) as source,destination.open('xb') as out:
                    shutil.copyfileobj(source,out,length=1024*1024)
                assert destination.stat().st_size==member.size
        assert not any('test' in p.name.lower() for p in extracted.iterdir())
        report['archive_members']=members
        times['safe_train_extraction_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        train=extracted/'train.bin';assert train.stat().st_size==50000*3074
        records=np.memmap(train,mode='r',dtype=np.uint8,shape=(50000,3074))
        fine=np.asarray(records[:,1]);coarse=np.asarray(records[:,0])
        assert np.array_equal(np.bincount(fine,minlength=100),np.full(100,500))
        assert int(coarse.min())==0 and int(coarse.max())==19
        fit=[];dev=[]
        for cls in range(100):
            indices=np.flatnonzero(fine==cls).tolist()
            indices.sort(key=lambda index:hashlib.sha256(f'{SEED}|{cls}|{index}'.encode('ascii')).digest())
            dev.extend(indices[:100]);fit.extend(indices[100:])
        fit=np.array(sorted(fit),dtype=np.int64);dev=np.array(sorted(dev),dtype=np.int64)
        assert len(fit)==40000 and len(dev)==10000
        assert np.intersect1d(fit,dev).size==0
        assert np.array_equal(np.sort(np.concatenate([fit,dev])),np.arange(50000))
        assert np.array_equal(np.bincount(fine[fit],minlength=100),np.full(100,400))
        assert np.array_equal(np.bincount(fine[dev],minlength=100),np.full(100,100))
        splits=ROOT/'splits';splits.mkdir()
        np.save(splits/'fit_indices.npy',fit,allow_pickle=False)
        np.save(splits/'development_indices.npy',dev,allow_pickle=False)
        report['split']={'seed_string':SEED,'method':'Per fine class, rank official TRAIN row indices by SHA256(seed|class|index); first 100 development, remaining 400 fit; store each final set sorted. No image statistics or outcomes used.','fit':40000,'development':10000,'fit_per_class':400,'development_per_class':100,'official_test_used':False}
        report['train_shape']=[50000,3074]
        report['train_format']='uint8: coarse label, fine label, R[1024], G[1024], B[1024]; image reshape(3,32,32) from columns 2 onward'
        times['train_labels_and_split_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        report['files']=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest_file(p)} for p in sorted(extracted.iterdir())+sorted(splits.iterdir())]
        times['train_and_split_hash_seconds']=time.perf_counter()-t
        report['status']='ready_train_only'
    except Exception as exc:
        report['status']='blocked';report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['timings_seconds']=times;report['total_wall_seconds']=time.perf_counter()-start
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (ROOT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
