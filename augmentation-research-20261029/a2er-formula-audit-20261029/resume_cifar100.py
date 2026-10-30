#!/usr/bin/env python3
"""Resume the interrupted official transfer using four standard byte ranges.

Preserve prefix and initial receipt; verify official archive hash before train-only
extraction. No test payload extraction or model work.
"""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request
import numpy as np
from prepare_cifar100 import ROOT, URL, MD5, OFFICIAL_PAGE, SEED, digest_file

TOTAL=168513733
ETAG='"a0b50c5-47ffe904ab8f2"'


def main():
    started=time.perf_counter();times={}
    prefix=ROOT/'cifar-100-binary.tar.gz.partial'
    assert prefix.exists() and 0<prefix.stat().st_size<TOTAL
    size=prefix.stat().st_size
    old=ROOT/'preparation.json'
    if old.exists():shutil.copyfile(old,ROOT/'preparation-initial-transfer.json')
    report={'status':'resuming','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_url':URL,'official_page':OFFICIAL_PAGE,'official_archive_md5':MD5,'preserved_prefix_bytes':size,'preserved_prefix_sha256':digest_file(prefix),'range_evidence':{'status':206,'content_range':f'bytes 0-0/{TOTAL}','etag':ETAG},'test_policy':'Archive retained opaque; only train.bin and label-name metadata extracted. Test payload not parsed or evaluated.','training':False}
    chunks=ROOT/'download-parts';chunks.mkdir(exist_ok=False)
    def part(index,lo,hi):
        t=time.perf_counter();p=chunks/f'{index:02d}-{lo}-{hi}.part';count=0
        req=urllib.request.Request(URL,headers={'Range':f'bytes={lo}-{hi}','If-Range':ETAG})
        with urllib.request.urlopen(req,timeout=45) as r,p.open('xb') as f:
            assert r.status==206 and r.headers.get('Content-Range')==f'bytes {lo}-{hi}/{TOTAL}'
            assert r.headers.get('ETag')==ETAG
            for block in iter(lambda:r.read(1024*1024),b''):
                count+=len(block);assert count<=hi-lo+1;f.write(block)
        assert count==hi-lo+1
        result={'index':index,'lo':lo,'hi':hi,'bytes':count,'seconds':time.perf_counter()-t,'sha256':digest_file(p),'file':str(p)}
        print(json.dumps({'range_complete':index,'MiB':count/2**20,'seconds':result['seconds']}),flush=True)
        return result
    try:
        t=time.perf_counter();remaining=TOTAL-size;step=(remaining+3)//4
        jobs=[(i,size+i*step,min(TOTAL-1,size+(i+1)*step-1)) for i in range(4) if size+i*step<TOTAL]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            parts=list(pool.map(lambda x:part(*x),jobs))
        report['parts']=parts;times['range_download_seconds']=time.perf_counter()-t
        t=time.perf_counter();assembled=ROOT/'cifar-100-binary.tar.gz.assembled'
        with assembled.open('xb') as out:
            for path in [prefix]+[Path(p['file']) for p in sorted(parts,key=lambda p:p['lo'])]:
                with path.open('rb') as f:shutil.copyfileobj(f,out,length=1024*1024)
        assert assembled.stat().st_size==TOTAL
        assert digest_file(assembled,'md5')==MD5,'official archive MD5 mismatch; extraction forbidden'
        archive=ROOT/'cifar-100-binary.tar.gz';assembled.rename(archive)
        report['archive_sha256']=digest_file(archive);report['archive_bytes']=TOTAL
        times['assemble_and_verify_seconds']=time.perf_counter()-t
        t=time.perf_counter();members=[]
        wanted={'cifar-100-binary/train.bin','cifar-100-binary/fine_label_names.txt','cifar-100-binary/coarse_label_names.txt'}
        known=wanted|{'cifar-100-binary/test.bin','cifar-100-binary'}
        outdir=ROOT/'train-only';outdir.mkdir(exist_ok=False)
        with tarfile.open(archive,'r:gz') as tar:
            entries=tar.getmembers()
            assert len({e.name for e in entries})==len(entries)
            for e in entries:
                name=e.name.rstrip('/');p=PurePosixPath(name)
                assert name in known and not p.is_absolute() and '..' not in p.parts
                assert e.isfile() or e.isdir();assert e.size<=160*1024*1024
                members.append({'name':name,'size':e.size,'extracted':name in wanted})
            for e in entries:
                if e.name not in wanted:continue
                p=outdir/PurePosixPath(e.name).name
                with tar.extractfile(e) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,length=1024*1024)
                assert p.stat().st_size==e.size
        report['archive_members']=members
        assert not any('test' in p.name for p in outdir.iterdir())
        times['safe_train_extraction_seconds']=time.perf_counter()-t
        t=time.perf_counter();train=outdir/'train.bin'
        assert train.stat().st_size==50000*3074
        records=np.memmap(train,mode='r',dtype=np.uint8,shape=(50000,3074));fine=np.asarray(records[:,1]);coarse=np.asarray(records[:,0])
        assert np.array_equal(np.bincount(fine,minlength=100),np.full(100,500))
        assert int(coarse.min())==0 and int(coarse.max())==19
        fit=[];dev=[]
        for c in range(100):
            ids=np.flatnonzero(fine==c).tolist()
            ids.sort(key=lambda i:hashlib.sha256(f'{SEED}|{c}|{i}'.encode('ascii')).digest())
            dev.extend(ids[:100]);fit.extend(ids[100:])
        fit=np.array(sorted(fit),dtype=np.int64);dev=np.array(sorted(dev),dtype=np.int64)
        assert len(fit)==40000 and len(dev)==10000 and np.intersect1d(fit,dev).size==0
        assert np.array_equal(np.sort(np.concatenate([fit,dev])),np.arange(50000))
        assert np.array_equal(np.bincount(fine[fit],minlength=100),np.full(100,400))
        assert np.array_equal(np.bincount(fine[dev],minlength=100),np.full(100,100))
        splits=ROOT/'splits';splits.mkdir(exist_ok=False)
        np.save(splits/'fit_indices.npy',fit,allow_pickle=False);np.save(splits/'development_indices.npy',dev,allow_pickle=False)
        times['train_labels_and_split_seconds']=time.perf_counter()-t
        report['split']={'seed_string':SEED,'method':'Per fine class, SHA256(seed|class|official TRAIN row index), first 100 development, remaining 400 fit; final arrays sorted.','fit':40000,'development':10000,'fit_per_class':400,'development_per_class':100,'official_test_used':False}
        report['train_format']='uint8 [50000,3074]: coarse label, fine label, then RGB planar image reshape(3,32,32)'
        t=time.perf_counter()
        report['files']=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest_file(p)} for p in sorted(outdir.iterdir())+sorted(splits.iterdir())]
        times['train_and_split_hash_seconds']=time.perf_counter()-t
        report['status']='ready_train_only'
    except BaseException as e:
        report['status']='blocked';report['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        report['timings_seconds']=times;report['total_resume_wall_seconds']=time.perf_counter()-started
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (ROOT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
