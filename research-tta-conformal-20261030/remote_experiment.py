"""Independent CIFAR100 experiment, for user's own Colab only.

Downloads official public data; does not access Drive or any user project files.
Training has hard step/image/time limits. The local controller releases runtime.
"""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import tarfile
import time
import traceback
import urllib.request

ROOT = Path('/content/tta_conformal_20261030')
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
EVENTS = OUT / 'events.jsonl'
ALLOCATED_AT = float(os.environ['TTA_ALLOCATED_AT'])
RATE = float(os.environ['TTA_VERIFIED_RATE'])
assert 0 <= RATE <= 3
assert os.environ.get('TTA_PROTOCOL_SHA256')
SOURCE_START = time.time()
STATE = {'formal_steps':0, 'formal_presentations':0, 'benchmark_steps':0,
         'benchmark_presentations':0, 'inference_views':0, 'gpu_allocation_start':ALLOCATED_AT,
         'protocol_sha256':os.environ['TTA_PROTOCOL_SHA256'], 'verified_rate_cu_per_hour':RATE}


def event(kind, **values):
    d = {'event':kind, 'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'allocation_elapsed_seconds':time.time()-ALLOCATED_AT, **values}
    with EVENTS.open('a') as f: f.write(json.dumps(d,allow_nan=False)+'\n')
    print(json.dumps(d,allow_nan=False), flush=True)


def save(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def watchdog():
    if time.time()-ALLOCATED_AT > 110*60:
        raise RuntimeError('RESOURCE_STOP: 110-minute computation deadline; reserve for download/release')
    if STATE['formal_steps']+STATE['benchmark_steps']>25240:
        raise RuntimeError('Training step cap exceeded')
    if STATE['formal_presentations']+STATE['benchmark_presentations']>3225600:
        raise RuntimeError('Training presentation cap exceeded')
    if STATE['inference_views']>220000:
        raise RuntimeError('Inference view cap exceeded')
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if size>2*1024**3: raise RuntimeError('Storage cap exceeded')


def data_and_split(np):
    archive=ROOT/'cifar-100-python.tar.gz'
    event('download_started',source='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
    req=urllib.request.Request('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',headers={'User-Agent':'TTA-Conformal-Research/1.0'})
    md5=hashlib.md5()
    with urllib.request.urlopen(req,timeout=120) as src,archive.open('wb') as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk:break
            dst.write(chunk);md5.update(chunk)
            if time.time()-SOURCE_START>600:raise RuntimeError('Download exceeds 10 minute cap')
    assert md5.hexdigest()=='eb9058c3a382ffc7106e4002c42a8d85', 'Official archive MD5 mismatch'
    # Read only named files; no arbitrary tar extraction.
    with tarfile.open(archive,'r:gz') as tf:
        tr=pickle.load(tf.extractfile('cifar-100-python/train'),encoding='bytes')
        te=pickle.load(tf.extractfile('cifar-100-python/test'),encoding='bytes')
    x=np.concatenate([tr[b'data'],te[b'data']]).reshape(60000,3,32,32)
    y=np.asarray(tr[b'fine_labels']+te[b'fine_labels'],dtype=np.int64)
    hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in x]
    groups=collections.defaultdict(list)
    for i,h in enumerate(hashes):groups[h].append(i)
    rng=np.random.default_rng(20261030)
    seen=set(); train=[];train_hashes=set()
    for i in rng.permutation(50000):
        h=hashes[i]
        if h in seen:continue
        seen.add(h);g=groups[h]
        if max(g)>=50000:continue
        if len(train)+len(g)<=40000:train.extend(g);train_hashes.add(h)
    if len(train)!=40000:raise RuntimeError('SPLIT_STOP: cannot assign exact 40000 train without splitting duplicate groups')
    held=np.array([i for i in range(60000) if hashes[i] not in train_hashes],dtype=np.int64)
    assert len(held)==20000
    roles={'train':np.array(train,dtype=np.int64)}
    caps={'pseudocal':2000,'validation':2000,'cal':6000,'eval':10000}
    buffers={k:[] for k in caps};seen=set()
    for i in held[rng.permutation(20000)]:
        h=hashes[i]
        if h in seen:continue
        seen.add(h);g=groups[h]
        placed=False
        for name,cap in caps.items():
            if len(buffers[name])+len(g)<=cap:
                buffers[name].extend(g);placed=True;break
        if not placed:raise RuntimeError('SPLIT_STOP: duplicate groups prevent exact heldout role counts')
    for name,ids in buffers.items():
        assert len(ids)==caps[name]
        roles[name]=np.array(ids,dtype=np.int64)
    assignment={int(i):name for name,ids in roles.items() for i in ids}
    assert len(assignment)==60000
    assert all(len({assignment[i] for i in g})==1 for g in groups.values())
    with (OUT/'split-manifest.jsonl').open('w') as f:
        for i in range(60000):
            f.write(json.dumps({'id':i,'source':'official_train' if i<50000 else 'official_test',
                                'source_index':i if i<50000 else i-50000,'image_sha256':hashes[i],
                                'role':assignment[i]})+'\n')
    np.savez(OUT/'split-ids.npz',**roles)
    summary={'seed':20261030,'archive_md5':md5.hexdigest(),'archive_sha256':sha(archive),
             'duplicate_group_count':sum(len(g)>1 for g in groups.values()),
             'duplicate_images':sum(len(g) for g in groups.values() if len(g)>1),
             'cross_origin_duplicate_groups':sum(min(g)<50000<=max(g) for g in groups.values()),
             'cross_role_duplicates':0,'roles':{k:{'n':len(v),'official_train_n':int((v<50000).sum()),
                   'official_test_n':int((v>=50000).sum()),'ids_sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in roles.items()}}
    save('split-summary.json',summary)
    event('split_validated',**summary)
    return x,y,roles


def run():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    assert torch.cuda.is_available(), 'RESOURCE_STOP: no CUDA GPU'
    gpu=torch.cuda.get_device_name(0)
    assert 'T4' in gpu, 'RESOURCE_STOP: actual GPU does not match requested T4'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    metadata={'gpu':gpu,'torch':torch.__version__,'numpy':np.__version__,
              'python':sys.version,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'total_gpu_memory':torch.cuda.get_device_properties(0).total_memory,
              'code_sha256':sha(Path(__file__)),**STATE}
    save('environment.json',metadata)
    event('gpu_verified',**metadata)
    xcpu,ycpu,roles=data_and_split(np)
    watchdog()

    class Block(nn.Module):
        def __init__(self,cin,cout,stride):
            super().__init__();self.stride=stride;self.cin=cin;self.cout=cout
            self.c1=nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False)
            self.b1=nn.BatchNorm2d(cout);self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False)
            self.b2=nn.BatchNorm2d(cout)
        def forward(self,x):
            residual=x
            if self.stride!=1:
                residual=x[:,:,::2,::2]
                pad=(self.cout-self.cin)//2
                residual=F.pad(residual,(0,0,0,0,pad,pad))
            z=F.relu(self.b1(self.c1(x)))
            return F.relu(self.b2(self.c2(z))+residual)

    class ResNet20(nn.Module):
        def __init__(self):
            super().__init__();self.stem=nn.Conv2d(3,16,3,padding=1,bias=False);self.bn=nn.BatchNorm2d(16)
            blocks=[];cin=16
            for stage,cout in enumerate((16,32,64)):
                for b in range(3):
                    blocks.append(Block(cin,cout,2 if stage>0 and b==0 else 1));cin=cout
            self.blocks=nn.Sequential(*blocks);self.fc=nn.Linear(64,100)
            for m in self.modules():
                if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
                elif isinstance(m,nn.Linear):nn.init.normal_(m.weight,0,.01);nn.init.zeros_(m.bias)
        def forward(self,x):
            x=F.relu(self.bn(self.stem(x)));x=self.blocks(x)
            return self.fc(x.mean((2,3)))

    def seed():
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)

    t0=time.perf_counter()
    # Moving the full data once is included in benchmark overhead.
    x=torch.from_numpy(xcpu).to('cuda');y=torch.from_numpy(ycpu).to('cuda')
    trainidx=torch.from_numpy(roles['train']).to('cuda')
    means=[];squares=[]
    for chunk in trainidx.split(1000):
        z=x[chunk].to(torch.float64)/255
        means.append(z.sum((0,2,3)));squares.append((z*z).sum((0,2,3)))
    count=40000*32*32
    mean=torch.stack(means).sum(0)/count
    std=(torch.stack(squares).sum(0)/count-mean**2).sqrt()
    mean=mean.to(torch.float32)[None,:,None,None];std=std.to(torch.float32)[None,:,None,None]
    save('normalization.json',{'mean':mean.flatten().cpu().tolist(),'std':std.flatten().cpu().tolist(),'source':'40000 training images only'})
    del means,squares,z
    rows=torch.arange(32,device='cuda')[None,:,None]
    cols=torch.arange(32,device='cuda')[None,None,:]

    def augment(ids):
        z=F.pad(x[ids],(4,4,4,4))
        offsets=torch.randint(9,(len(ids),2),device='cuda')
        rr=rows+offsets[:,0,None,None];cc=cols+offsets[:,1,None,None]
        z=z.permute(0,2,3,1)[torch.arange(len(ids),device='cuda')[:,None,None],rr,cc].permute(0,3,1,2)
        flip=torch.rand(len(ids),device='cuda')<.5
        z=torch.where(flip[:,None,None,None],z.flip(-1),z)
        return (z.to(torch.float32)/255-mean)/std

    def model_optim():
        seed();m=ResNet20().cuda()
        opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
        return m,opt

    def update(m,opt,ids):
        opt.zero_grad(set_to_none=True)
        logits=m(augment(ids));loss=F.cross_entropy(logits,y[ids]);loss.backward();opt.step()
        return loss.detach()

    def infer(m,ids,dest):
        mm=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(ids),10,100))
        m.eval();offsets=[(0,0),(0,8),(8,0),(8,8),(4,4)]
        written_hash=hashlib.sha256()
        with torch.inference_mode():
            for begin in range(0,len(ids),128):
                idbatch=ids[begin:begin+128]
                z=F.pad(x[idbatch],(4,4,4,4));views=[z[:,:,a:a+32,b:b+32] for a,b in offsets]
                assert torch.equal(views[4],x[idbatch])
                views=views+[v.flip(-1) for v in views]
                # Ten views in one GPU call; fixed image-major axis restored afterward.
                inp=torch.stack(views,dim=1).reshape(-1,3,32,32).to(torch.float32)/255
                logits=m((inp-mean)/std).reshape(len(idbatch),10,100).cpu().numpy()
                assert logits.dtype==np.float32 and np.isfinite(logits).all()
                written_hash.update(logits.tobytes(order='C'))
                mm[begin:begin+len(idbatch)]=logits
                STATE['inference_views']+=len(idbatch)*10
                if begin%1280==0:watchdog()
        mm.flush();del mm
        rd=np.load(dest,mmap_mode='r')
        assert rd.shape==(len(ids),10,100) and rd.dtype==np.float32
        read_hash=hashlib.sha256()
        for j in range(0,len(ids),128):read_hash.update(rd[j:j+128].tobytes(order='C'))
        assert written_hash.hexdigest()==read_hash.hexdigest(), 'Logit write/read bitwise hash mismatch'
        return rd

    m,opt=model_optim();m.train();torch.cuda.synchronize()
    setup_s=time.perf_counter()-t0
    segment=[];last=time.perf_counter();perm=trainidx[torch.randperm(40000,device='cuda')]
    for step in range(200):
        loss=update(m,opt,perm[step*128:(step+1)*128])
        STATE['benchmark_steps']+=1;STATE['benchmark_presentations']+=128
        if (step+1)%50==0:
            torch.cuda.synchronize();now=time.perf_counter();segment.append((now-last)/50);last=now
            event('benchmark_training',updates=step+1,segment_seconds_per_step=segment[-1]);watchdog()
    b_ids=torch.from_numpy(np.concatenate([roles[k] for k in ('pseudocal','validation','cal','eval')])[:512]).to('cuda')
    torch.cuda.synchronize();t0=time.perf_counter()
    bench=infer(m,b_ids,OUT/'benchmark-logits.npy')
    bench_copy=np.array(bench);del bench
    assert np.array_equal(np.load(OUT/'benchmark-logits.npy'),bench_copy)
    assert np.array_equal(bench_copy.mean(1,dtype=np.float64),np.load(OUT/'benchmark-logits.npy').mean(1,dtype=np.float64))
    torch.cuda.synchronize();inference_s=time.perf_counter()-t0
    # Retain digest, discard pilot data and pilot weights.
    bench_hash=sha(OUT/'benchmark-logits.npy');(OUT/'benchmark-logits.npy').unlink()
    del m,opt,bench_copy,loss;torch.cuda.empty_cache()
    train_prediction=1.5*max(segment)*25040
    infer_prediction=1.5*inference_s/512*20000
    elapsed=time.time()-ALLOCATED_AT
    predicted=elapsed+train_prediction+infer_prediction+240
    perf={'train_segments_seconds_per_step':segment,'setup_transfer_normalization_seconds':setup_s,
          'benchmark_inference_images':512,'benchmark_inference_seconds':inference_s,
          'benchmark_logit_sha256':bench_hash,'weights_discarded':True,
          'conservative_remaining_training_seconds':train_prediction,
          'conservative_remaining_inference_seconds':infer_prediction,
          'download_release_allowance_seconds':240,'predicted_total_gpu_seconds':predicted,
          'predicted_cu':predicted/3600*RATE,'pass':predicted<=5400 and predicted/3600*RATE<=6}
    save('performance-gate.json',perf);event('performance_gate',**perf)
    if not perf['pass']:
        STATE['result']='RESOURCE_STOP_PERFORMANCE';return
    watchdog()

    m,opt=model_optim();m.train();start=time.perf_counter()
    for epoch in range(80):
        perm=trainidx[torch.randperm(40000,device='cuda')]
        losses=[]
        for begin in range(0,40000,128):
            step=STATE['formal_steps'];lr=.1*.5*(1+math.cos(math.pi*step/25040))
            for group in opt.param_groups:group['lr']=lr
            ids=perm[begin:begin+128];loss=update(m,opt,ids);losses.append(loss)
            STATE['formal_steps']+=1;STATE['formal_presentations']+=len(ids)
            if STATE['formal_steps']%100==0:watchdog()
        torch.cuda.synchronize()
        event('epoch_done',epoch=epoch+1,formal_steps=STATE['formal_steps'],
              train_loss=float(torch.stack(losses).mean().cpu()),lr=lr,
              training_seconds=time.perf_counter()-start)
    assert STATE['formal_steps']==25040 and STATE['formal_presentations']==3200000
    torch.save({'state_dict':m.state_dict(),'seed':0,'epochs':80,'steps':25040,
                'architecture':'independent CIFARResNet20 option A','normalization':json.loads((OUT/'normalization.json').read_text())},OUT/'final-model.pt')
    held_ids=np.concatenate([roles[k] for k in ('pseudocal','validation','cal','eval')])
    hid=torch.from_numpy(held_ids).to('cuda')
    np.save(OUT/'heldout-image-ids.npy',held_ids)
    np.save(OUT/'heldout-labels.npy',ycpu[held_ids])
    np.save(OUT/'view-ids.npy',np.arange(10,dtype=np.int64))
    event('formal_inference_started',images=20000,view_count=10)
    logits=infer(m,hid,OUT/'logits-fp32.npy')
    assert STATE['inference_views']==205120
    # Independent readback and streaming mean verification, no extra GPU views.
    mean_path=OUT/'mean-logits-f64.npy'
    means=np.lib.format.open_memmap(mean_path,mode='w+',dtype=np.float64,shape=(20000,100))
    for i in range(0,20000,512):means[i:i+512]=logits[i:i+512].mean(1,dtype=np.float64)
    means.flush();del means;check=np.load(mean_path,mmap_mode='r')
    np.save(OUT/'baseline-logits-fp32.npy',logits[:,4,:])
    rd=np.load(OUT/'logits-fp32.npy',mmap_mode='r')
    for i in range(0,20000,512):
        assert np.array_equal(logits[i:i+512],rd[i:i+512])
        assert np.array_equal(check[i:i+512],rd[i:i+512].mean(1,dtype=np.float64))
    assert np.array_equal(np.load(OUT/'baseline-logits-fp32.npy'),rd[:,4,:])
    save('cache-acceptance.json',{'pass':True,'shape':list(rd.shape),'dtype':str(rd.dtype),
         'readback_bitwise_equal':True,'means_float64_equal':True,'baseline_view':4,
         'id_unique_count':len(np.unique(held_ids)),'image_ids_sha256':sha(OUT/'heldout-image-ids.npy'),
         'view_ids_sha256':sha(OUT/'view-ids.npy'),'logits_sha256':sha(OUT/'logits-fp32.npy')})
    STATE['result']='LOGITS_COMPLETE'
    event('logits_complete',**STATE)


try:
    run()
except BaseException as exc:
    STATE['result']='ERROR_OR_RESOURCE_STOP';STATE['error_type']=type(exc).__name__;STATE['error']=str(exc)
    (OUT/'error.txt').write_text(traceback.format_exc())
    event('stopped',error_type=type(exc).__name__,error=str(exc))
finally:
    STATE['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    STATE['allocation_elapsed_seconds_at_script_end']=time.time()-ALLOCATED_AT
    save('run-state.json',STATE)
    hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    save('artifact-hashes.json',hashes)
    bundle=ROOT/'results.tar'
    with tarfile.open(bundle,'w') as tf:
        for p in sorted(OUT.iterdir()):
            if p.is_file():tf.add(p,arcname=p.name)
    print(json.dumps({'BUNDLE_READY':str(bundle),'bytes':bundle.stat().st_size,'sha256':sha(bundle),
                      'result':STATE.get('result')}),flush=True)
