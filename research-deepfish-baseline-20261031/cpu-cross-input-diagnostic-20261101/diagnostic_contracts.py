"""Pure identity, atomic persistence, resource and original COCO contracts.

Importing this module imports no model or model framework.
"""
import collections
import errno
import hashlib
import json
import math
import os
import pathlib
import shutil
import time

P = pathlib.Path(__file__).resolve().parent
R = P.parent
PREP = R/'cpu-cross-input-preflight-20261101'
PROTOCOL_SHA = '0b4101c0e9e41904c8dc23618439fa2499815eb136ea106bd3c48ef1ed02e305'
FACTORS = [('c100', 1.0), ('c080', 0.8)]
METRIC_MAPPING = [
    ('AP',0,100,'all'),('AP50',1,1000,'all'),('AP75',2,1000,'all'),
    ('AP_small',3,1000,'small'),('AP_medium',4,1000,'medium'),('AP_large',5,1000,'large'),
    ('AR100',6,100,'all'),('AR300',7,300,'all'),('AR1000',8,1000,'all'),
    ('AR_small1000',9,1000,'small'),('AR_medium1000',10,1000,'medium'),('AR_large1000',11,1000,'large'),
]

class ContractError(RuntimeError): pass
class ResourceStop(RuntimeError): pass

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def atomic_json(path,value,*,new=False):
    """File fsync + same-directory replace + directory fsync, then readback hash.

    An exact existing new record is idempotent; a conflicting record is rejected.
    Transient EINTR/EAGAIN during publication retries the same bytes, never inference.
    """
    path=pathlib.Path(path);data=canonical(value)+b'\n'
    if new and path.exists():
        if path.read_bytes()!=data:raise ContractError('Conflicting existing record: '+str(path))
        return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'existing_exact':True}
    tmp=path.with_name(path.name+'.partial')
    if tmp.exists():
        if tmp.read_bytes()!=data:raise ContractError('Unresolved partial file: '+str(tmp))
    else:
        with tmp.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
    for attempt in range(3):
        try:os.replace(tmp,path);break
        except OSError as exc:
            if exc.errno not in (errno.EINTR,errno.EAGAIN) or attempt==2:raise
            time.sleep(.05)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    actual=path.read_bytes()
    if actual!=data:raise ContractError('Atomic persistence readback differs')
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'existing_exact':False}

def identity(seed,arm,condition,image_id):
    if seed not in (20261031,20261101) or arm not in ('A','B') or condition not in ('c100','c080'):
        raise ContractError('Identity outside frozen matrix')
    if not isinstance(image_id,int):raise ContractError('Image ID must be integer')
    return f'{seed}_{arm}_{condition}_{image_id}'

def expected_ids(models,image_ids):
    values=[identity(m['seed'],m['arm'],c,i) for m in models for c,_ in FACTORS for i in image_ids]
    require_complete_ids(values,values)
    return values

def require_complete_ids(actual,expected):
    ac=collections.Counter(actual);ex=collections.Counter(expected)
    if len(expected)!=len(set(expected)):raise ContractError('Duplicate expected identities')
    if any(v!=1 for v in ac.values()):raise ContractError('Duplicate completed identities')
    if ac!=ex:raise ContractError(f'Incomplete identity matrix: missing={len(ex-ac)}, extra={len(ac-ex)}')

def read_protocol():
    if sha(PREP/'PROTOCOL.json')!=PROTOCOL_SHA:raise ContractError('Original protocol changed')
    return json.loads((PREP/'PROTOCOL.json').read_text())

def read_execution():
    run=json.loads((P/'RUN_PROTOCOL.json').read_text())
    if run['original_protocol_sha256']!=PROTOCOL_SHA or not run['authorized_full512']:
        raise ContractError('Formal authorization/protocol mismatch')
    return run

def check_payload(payload,expected_id,run):
    if payload['identity']!=expected_id:raise ContractError('Record identity mismatch')
    manifest=run['identity_map'][expected_id]
    for k in ('seed','arm','condition','image_id','factor','checkpoint_sha256','config_sha256','image_sha256'):
        if payload[k]!=manifest[k]:raise ContractError('Record provenance differs: '+k)
    if payload['annotation_sha256']!=run['dev_annotation']['sha256']:
        raise ContractError('Annotation differs')
    if payload['environment_lock_sha256']!=run['environment_lock_sha256']:
        raise ContractError('Environment differs')
    preds=payload['predictions']
    if len(preds)!=300:raise ContractError('Prediction count differs from full official top300')
    for row in preds:
        if row['image_id']!=manifest['image_id'] or row['category_id']!=1:
            raise ContractError('COCO image/category identity mismatch')
        b=row['bbox'];s=row['score']
        if len(b)!=4 or not all(math.isfinite(v) for v in b+[s]):raise ContractError('Nonfinite prediction')
        if b[2]<0 or b[3]<0 or not 0<=s<=1:raise ContractError('Invalid unmodified detector output')
    return payload

def persist_prediction(payload,run):
    key=payload['identity'];check_payload(payload,key,run)
    envelope={'payload_sha256':digest(payload),'payload':payload}
    return atomic_json(P/'predictions'/(key+'.json'),envelope,new=True)

def load_prediction(path,run):
    envelope=json.loads(pathlib.Path(path).read_text())
    payload=envelope['payload']
    if envelope['payload_sha256']!=digest(payload):raise ContractError('Persisted payload hash differs')
    check_payload(payload,pathlib.Path(path).stem,run)
    return payload

def complete_matrix(run):
    files=sorted((P/'predictions').glob('*.json'))
    require_complete_ids([f.stem for f in files],run['expected_identities'])
    if list((P/'predictions').glob('*.partial')):raise ContractError('Unresolved partial prediction files')
    entries=[];geometry={}
    for path in files:
        payload=load_prediction(path,run);image_id=payload['image_id']
        old=geometry.setdefault(image_id,payload['geometry_sha256'])
        if old!=payload['geometry_sha256']:raise ContractError('Geometry/GT changed across matrix')
        entries.append({'identity':payload['identity'],'path':str(path.relative_to(P)),
                        'bytes':path.stat().st_size,'sha256':sha(path)})
    if len(geometry)!=64:raise ContractError('Expected exactly64 dev images')
    return {'status':'ALL_512_UNIQUE_COMPLETE','records':entries,'manifest_sha256':digest(entries),
            'images':64,'conditions':8,'records_count':512,'geometry_GT_same_across8':True}

def tree_bytes(root):return sum(f.stat().st_size for f in pathlib.Path(root).rglob('*') if f.is_file())

class Guard:
    def __init__(self,context):
        self.context=context;self.last_full=0;self.other_bytes=0
    def check(self,extra=0):
        now=time.monotonic()
        if now>=self.context['deadline_monotonic']:raise ResourceStop('Formal90min wall-time limit reached')
        produced=tree_bytes(P)
        if now-self.last_full>30:
            self.other_bytes=tree_bytes(R)-produced;self.last_full=now
        if produced+extra>100*1024**2:raise ResourceStop('New formal artifacts100MiB limit reached')
        if self.other_bytes+produced+extra>6*1024**3-80*1024**2:
            raise ResourceStop('Research6GiB cap / retained80MiB margin reached')
        if shutil.disk_usage(R).free<80*1024**2+extra:raise ResourceStop('Filesystem safety margin reached')
        return {'formal_bytes':produced,'research_bytes_estimate':self.other_bytes+produced}

def mapped_stats(stats):
    if len(stats)!=12 or not all(math.isfinite(float(v)) for v in stats):
        raise ContractError('Unexpected COCO statistics')
    return {name:{'value':float(stats[index]),'percent':float(stats[index])*100,
                  'maxDets':maxdets,'area':area,'stats_index':index}
            for name,index,maxdets,area in METRIC_MAPPING}

def array_digest(array):
    h=hashlib.sha256(canonical([str(array.dtype),list(array.shape)]));h.update(array.tobytes())
    return h.hexdigest()

def evaluate_condition(run,model,condition):
    """Official COCOeval, original MMDetection parameter mapping, persisted rows only."""
    import copy
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rows=[]
    for image_id in run['image_order']:
        key=identity(model['seed'],model['arm'],condition,image_id)
        rows.extend(load_prediction(P/'predictions'/(key+'.json'),run)['predictions'])
    if len(rows)!=64*300:raise ContractError('Incomplete condition')
    if sha(run['dev_annotation']['path'])!=run['dev_annotation']['sha256']:
        raise ContractError('Dev annotation changed before evaluation')
    gt=COCO(run['dev_annotation']['path'])
    dt=gt.loadRes(copy.deepcopy(rows));ev=COCOeval(gt,dt,'bbox')
    ev.params.catIds=[1];ev.params.imgIds=list(run['image_order'])
    ev.params.maxDets=[100,300,1000]
    ev.params.iouThrs=np.linspace(.5,.95,10,endpoint=True)
    ev.evaluate();ev.accumulate();ev.summarize()
    arrays={k:ev.eval[k] for k in ('precision','recall','scores')}
    hashes={k:array_digest(v) for k,v in arrays.items()}
    return {'seed':model['seed'],'arm':model['arm'],'condition':condition,
            'stats':[float(x) for x in ev.stats],'metrics':mapped_stats(ev.stats),
            'array_sha256':hashes,'predictions':len(rows),'images':64,
            'maxDets':[100,300,1000],'iouThrs':ev.params.iouThrs.tolist()},arrays
