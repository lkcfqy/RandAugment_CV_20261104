import collections
import concurrent.futures
import importlib.metadata
import json
import time
import numpy as np
from PIL import Image, ImageOps
import imagehash
from skimage.metrics import structural_similarity
from .protocol import ASSETS,WORK,SPEC,SEEDS,digest,save_json,file_sha

def read_rgb(path,size=224):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)/np.float32(255)

def _fingerprint(record):
    with Image.open(ASSETS/'data'/record['path']) as im:
        rgb=ImageOps.exif_transpose(im).convert('RGB')
        ph=int(str(imagehash.phash(rgb,hash_size=8,highfreq_factor=4)),16)
        gray=np.asarray(rgb.convert('L').resize((256,256),Image.Resampling.LANCZOS),dtype=np.uint8)
    return ph,gray

def is_near(ph1,g1,ph2,g2):
    hd=(int(ph1)^int(ph2)).bit_count()
    if hd>4:return False,hd,None
    value=float(structural_similarity(g1,g2,data_range=255,win_size=7,gaussian_weights=False,use_sample_covariance=True,K1=.01,K2=.03))
    return value>=.995,hd,value

class Components:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def build_split_manifest():
    started=time.monotonic()
    records=[json.loads(line) for line in (ASSETS/'reports/file-manifest.jsonl').read_text().splitlines()]
    records=[r for r in records if r['split'] in ('train','test')]
    byhash=collections.defaultdict(list)
    for r in records:byhash[r['sha256_decoded_rgb']].append(r)
    excluded=[];conflicts=[]
    for h,group in byhash.items():
        if {x['split'] for x in group}=={'train','test'}:
            labels={x['label']=='good' for x in group}
            if len(labels)!=1:conflicts.append([x['path'] for x in group])
            else:excluded.extend(x['path'] for x in group if x['split']=='train')
    if conflicts:
        save_json(WORK/'reports/data-conflict.json',{'status':'stop_label_conflict','conflicts':conflicts})
        raise RuntimeError('Train/test exactRGB label conflict; no computation authorized')
    train_groups=[]
    for h,group in sorted(byhash.items()):
        members=[x for x in group if x['split']=='train' and x['path'] not in excluded]
        if members:train_groups.append(sorted(members,key=lambda x:(x['sha256_decoded_rgb'],x['path'])))
    tests=sorted([r for r in records if r['split']=='test'],key=lambda r:(['same','illumination'].index(r['domain']),r['label']!='good',r['path']))
    canonical=[g[0] for g in train_groups]
    fp_records=canonical+tests
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:fp=list(pool.map(_fingerprint,fp_records))
    train_fp=fp[:len(canonical)];test_fp=fp[len(canonical):]
    components=Components(len(canonical));train_edges=[];cross_edges=[];candidates=0
    for i in range(len(canonical)):
        for j in range(i):
            if (train_fp[i][0]^train_fp[j][0]).bit_count()>4:continue
            candidates+=1;near,hd,value=is_near(*train_fp[i],*train_fp[j])
            if near:components.union(i,j);train_edges.append({'a':canonical[j]['path'],'b':canonical[i]['path'],'hamming':hd,'ssim':value})
    for i,r in enumerate(canonical):
        for j,t in enumerate(tests):
            if (train_fp[i][0]^test_fp[j][0]).bit_count()>4:continue
            near,hd,value=is_near(*train_fp[i],*test_fp[j])
            if near:cross_edges.append({'train':r['path'],'test':t['path'],'hamming':hd,'ssim':value,'action':'report_only'})
    groups=collections.defaultdict(list)
    for i,g in enumerate(train_groups):groups[components.find(i)].extend(g)
    clusters=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:(r['sha256_decoded_rgb'],r['path']))
        members=[{'pixelhash':r['sha256_decoded_rgb'],'path':r['path']} for r in group]
        cid=digest('\n'.join(m['pixelhash']+'\t'+m['path'] for m in members))
        clusters.append({'cluster_id':cid,'canonical':members[0],'members':members,'exact_unique_count':len({m['pixelhash'] for m in members})})
    clusters=sorted(clusters,key=lambda c:(c['canonical']['pixelhash'],c['canonical']['path']))
    cal=sorted(clusters,key=lambda c:(digest('stheta-v1/cal/'+c['cluster_id']),c['cluster_id']))[:64]
    cal_ids={c['cluster_id'] for c in cal};eligible=[c for c in clusters if c['cluster_id'] not in cal_ids]
    ready=len(cal)==64 and len(eligible)>=64
    refs={}
    if ready:
        for seed in SEEDS:
            chosen=sorted(eligible,key=lambda c:(digest(f'stheta-v1/refs/{seed}/'+c['cluster_id']),c['cluster_id']))[:64]
            refs[str(seed)]=sorted(chosen,key=lambda c:(c['canonical']['pixelhash'],c['canonical']['path']))
    assert sum(t['domain']=='same' and t['label']=='good' for t in tests)==230
    assert sum(t['domain']=='same' and t['label']!='good' for t in tests)==459
    assert sum(t['domain']=='illumination' and t['label']=='good' for t in tests)==75
    assert sum(t['domain']=='illumination' and t['label']!='good' for t in tests)==198
    result={'status':'ready' if ready else 'data_unresolved_insufficient_clusters','definition':SPEC['near_duplicate'],
      'source_file_manifest_sha256':file_sha(ASSETS/'reports/file-manifest.jsonl'),'train_original_paths':sum(r['split']=='train' for r in records),
      'exact_unique_train_nodes':len(train_groups),'cluster_count':len(clusters),'excluded_train_exact_crosssplit_paths':excluded,
      'train_near_edges':train_edges,'train_phash_candidate_pairs':candidates,'cross_split_near_edges_report_only':cross_edges,
      'clusters':clusters,'calibration':cal,'references':refs,
      'tests':[{'path':r['path'],'domain':r['domain'],'label':int(r['label']!='good'),'pixelhash':r['sha256_decoded_rgb']} for r in tests],
      'fingerprints':[{'path':r['path'],'phash64':f'{p[0]:016x}'} for r,p in zip(fp_records,fp)],
      'physical_entity_ids':None,'independence_claim':False,'versions':{p:importlib.metadata.version(p) for p in ['ImageHash','Pillow','scikit-image','numpy','scipy']},'elapsed_seconds':time.monotonic()-started}
    save_json(WORK/'split-manifest.json',result)
    print(json.dumps({k:result[k] for k in ['status','train_original_paths','exact_unique_train_nodes','cluster_count','elapsed_seconds']}),flush=True)
    if not ready:raise RuntimeError('Insufficient train clusters for fixed 64 cal + 64 refs; do not reduce protocol')
    return result
