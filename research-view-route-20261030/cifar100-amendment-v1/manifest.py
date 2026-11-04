"""CIFAR-100 original archive parser and fixed splits. Never execute pickle globals."""
import hashlib,json,pickletools,struct,tarfile,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT.parent.parent/'research-tta-conformal-20261030'/'recovered-data'/'cifar-100-python.tar.gz'
MD5='eb9058c3a382ffc7106e4002c42a8d85';SHA='85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
def sha(p,alg='sha256'):
    with p.open('rb') as f:return hashlib.file_digest(f,alg).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
MARK=object()
def inert_pickle(blob):
    """An inert opcode interpreter, narrowly validating original CIFAR ndarray state.

    GLOBAL creates a string tag, REDUCE/BUILD never invoke deserialized callables.
    No pickle.load, eval, torch allowlist, or imported remote globals is used.
    """
    stack=[];memo={}
    def marked():
        i=len(stack)-1
        while stack[i] is not MARK:i-=1
        values=stack[i+1:];del stack[i:];return values
    for op,arg,pos in pickletools.genops(blob):
        name=op.name
        if name=='PROTO':assert arg==2
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='MARK':stack.append(MARK)
        elif name in ['BINPUT','LONG_BINPUT']:memo[arg]=stack[-1]
        elif name in ['BINGET','LONG_BINGET']:stack.append(memo[arg])
        elif name in ['SHORT_BINSTRING','BINSTRING','BINUNICODE','UNICODE']:stack.append(arg)
        elif name in ['BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4']:stack.append(arg)
        elif name=='NONE':stack.append(None)
        elif name=='NEWFALSE':stack.append(False)
        elif name=='NEWTRUE':stack.append(True)
        elif name=='TUPLE':stack.append(tuple(marked()))
        elif name in ['TUPLE1','TUPLE2','TUPLE3']:
            n=int(name[-1]);v=tuple(stack[-n:]);del stack[-n:];stack.append(v)
        elif name=='APPENDS':v=marked();stack[-1].extend(v)
        elif name=='APPEND':v=stack.pop();stack[-1].append(v)
        elif name=='SETITEMS':
            v=marked();assert len(v)%2==0
            for i in range(0,len(v),2):stack[-1][v[i]]=v[i+1]
        elif name=='SETITEM':v=stack.pop();k=stack.pop();stack[-1][k]=v
        elif name=='GLOBAL':
            assert arg in ['numpy.core.multiarray _reconstruct','numpy ndarray','numpy dtype'],('Rejected global',arg)
            stack.append(('INERT_GLOBAL',arg))
        elif name=='REDUCE':
            args=stack.pop();fn=stack.pop()
            if fn==('INERT_GLOBAL','numpy.core.multiarray _reconstruct'):
                assert args[0]==('INERT_GLOBAL','numpy ndarray') and args[1]==(0,)
                stack.append({'INERT_ARRAY':True})
            elif fn==('INERT_GLOBAL','numpy dtype'):
                assert args==('u1',0,1);stack.append({'INERT_DTYPE':'u1'})
            else:raise ValueError(('Rejected reduce',fn))
        elif name=='BUILD':
            state=stack.pop();obj=stack[-1]
            if isinstance(obj,dict) and 'INERT_DTYPE' in obj:
                assert state==(3,'|',None,None,None,-1,-1,0)
            elif isinstance(obj,dict) and 'INERT_ARRAY' in obj:
                version,shape,dtype,fortran,raw=state
                assert version==1 and shape in [(50000,3072),(10000,3072)] and dtype=={'INERT_DTYPE':'u1'} and not fortran
                raw=raw.encode('latin1') if isinstance(raw,str) else raw
                assert len(raw)==shape[0]*shape[1]
                stack[-1]=np.frombuffer(raw,dtype=np.uint8).reshape(shape)
            else:raise ValueError('Rejected build')
        elif name=='STOP':
            assert len(stack)==1;return stack[0]
        else:raise ValueError(('Rejected opcode',name,pos))
    raise ValueError('No STOP opcode')

def main():
    began=time.time();assert not (ROOT/'data-manifest.json').exists(),'Never overwrite an existing split manifest'
    assert sha(ARCHIVE,'md5')==MD5 and sha(ARCHIVE)==SHA
    records=[];arrays=[];sourcefiles={}
    with tarfile.open(ARCHIVE,'r:gz') as tar:
        for split,n in [('train',50000),('test',10000)]:
            member='cifar-100-python/'+split;blob=tar.extractfile(member).read();sourcefiles[member]=hashlib.sha256(blob).hexdigest()
            obj=inert_pickle(blob);x=obj['data'];labels=obj['fine_labels'];names=obj['filenames']
            assert x.shape==(n,3072) and x.dtype==np.uint8 and len(labels)==n and len(names)==n
            assert Counter(labels)=={c:n//100 for c in range(100)}
            rgb=x.reshape(n,3,32,32).transpose(0,2,3,1).copy();arrays.append(rgb)
            offset=0 if split=='train' else 50000
            for i in range(n):
                ident=f'{split}/{i:05d}'
                records.append({'image_id':ident,'official_split':'trainval' if split=='train' else 'test','class_id':labels[i],
                    'original_filename':names[i],'source_member':member,'source_file_sha256':sourcefiles[member],'raw_array_index':offset+i,
                    'width':32,'height':32,'original_record_sha256':hashlib.sha256(x[i].tobytes()).hexdigest(),
                    'rgb_dimensions_sha256':hashlib.sha256(struct.pack('>II',32,32)+rgb[i].tobytes()).hexdigest()})
    np.save(ROOT/'data'/'original_rgb_uint8.npy',np.concatenate(arrays),allow_pickle=False)
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for field in ['original_record_sha256','rgb_dimensions_sha256']:
        seen={}
        for i,r in enumerate(records):
            if r[field] in seen:parent[find(i)]=find(seen[r[field]])
            else:seen[r[field]]=i
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[find(i)].append(r)
    duplicates=[]
    for group in groups.values():
        group=sorted(group,key=lambda x:x['image_id']);labels={x['class_id'] for x in group};test=[x for x in group if x['official_split']=='test']
        kept=None if len(labels)>1 else (test or group)[0]['image_id']
        for r in group:
            r['included']=r['image_id']==kept;r['exclusion_reason']=None if r['included'] else 'label_conflict' if len(labels)>1 else 'duplicate'
        if len(group)>1:duplicates.append({'ids':[x['image_id'] for x in group],'labels':sorted(labels),'kept':kept})
    val=[];supports={str(s):[] for s in [11,22,33]};test=[]
    for cls in range(100):
        pool=[r['image_id'] for r in records if r['included'] and r['official_split']=='trainval' and r['class_id']==cls]
        tp=[r['image_id'] for r in records if r['included'] and r['official_split']=='test' and r['class_id']==cls]
        def sorting(domain,seed=None):
            prefix=f'cifar100-route-v1|{domain}|'+(f'{seed}|' if seed is not None else '')+f'{cls}|'
            return lambda ident:(hashlib.sha256((prefix+ident).encode()).hexdigest(),ident)
        v=sorted(pool,key=sorting('validation'))[:16];assert len(v)==16;val+=v;pool=set(pool)-set(v)
        for seed in supports:
            chosen=sorted(pool,key=sorting('support',seed))[:16];assert len(chosen)==16;supports[seed]+=chosen
        chosen=sorted(tp,key=sorting('test'))[:50];assert len(chosen)==50;test+=chosen
    assert len(val)==1600 and len(test)==5000 and all(len(v)==1600 for v in supports.values())
    splits={'version':'cifar100-route-v1','class_id_encoding':'official fine labels 0..99','official_image_id_encoding':'official split + slash + zero-padded 5-digit record index',
            'validation':val,'support':supports,'test':test,'extra_validation_labels_per_class':16,'test_raw_data_read_only_for_hash_before_selection':True,
            'record_hash':'SHA256(original CHW uint8 row bytes); source batch-file SHA separately recorded; no per-image source files exist',
            'rgb_hash':'SHA256(big-endian uint32 width + height + original HWC RGB bytes); no EXIF metadata exists in raw CIFAR arrays'}
    dump(ROOT/'data-manifest.json',records);dump(ROOT/'splits.json',splits);dump(ROOT/'duplicates.json',duplicates)
    summary={'status':'passed','seconds':time.time()-began,'original_counts':{'train':50000,'test':10000},'included_counts':dict(Counter(r['official_split'] for r in records if r['included'])),
        'exclusion_counts':dict(Counter(r['exclusion_reason'] for r in records if not r['included'])),'duplicate_groups':len(duplicates),
        'validation_count':len(val),'support_per_seed':{s:len(x) for s,x in supports.items()},'test_count':len(test),
        'manifest_sha256':sha(ROOT/'data-manifest.json'),'splits_sha256':sha(ROOT/'splits.json'),'raw_uint8_npy_sha256':sha(ROOT/'data'/'original_rgb_uint8.npy'),
        'source_archive':str(ARCHIVE.resolve()),'source_archive_md5':MD5,'source_archive_sha256':SHA,'source_file_sha256':sourcefiles,
        'existing_archive_read_only':True,'previous_TTA_splits_or_logits_used':False,'raw_pickle_code_executed':False,'enlarged_images_cached':False}
    dump(ROOT/'data-validation.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
