"""Existing metadata/array inspection and prospective identities; no model imports."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import hashlib,json,collections,struct,time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=_portable_project_path(__file__, '')
DATA=PROJECT/'research-view-route-20261030/cifar100-amendment-v1'
ARCHIVE=PROJECT/'research-tta-conformal-20261030/recovered-data/cifar-100-python.tar.gz'
SALT='augmentation-selection-cifar100-20261102-v1'

def sha(p,algorithm='sha256'):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()
def key(*args):return hashlib.sha256((SALT+'|'+'|'.join(map(str,args))).encode()).hexdigest()
def dump(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def main():
    t=time.perf_counter()
    receipt=json.loads((DATA/'data-validation.json').read_text())
    assert sha(ARCHIVE)==receipt['source_archive_sha256']=='85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
    assert sha(ARCHIVE,'md5')==receipt['source_archive_md5']=='eb9058c3a382ffc7106e4002c42a8d85'
    assert sha(DATA/'data-manifest.json')==receipt['manifest_sha256']
    assert sha(DATA/'data/original_rgb_uint8.npy')==receipt['raw_uint8_npy_sha256']
    records=json.loads((DATA/'data-manifest.json').read_text())
    raw=np.load(DATA/'data/original_rgb_uint8.npy',mmap_mode='r',allow_pickle=False)
    assert raw.dtype==np.uint8 and raw.shape==(60000,32,32,3)
    assert len(records)==60000 and {r['class_id'] for r in records}==set(range(100))
    byid={r['image_id']:r for r in records};assert len(byid)==60000
    eligible=[r for r in records if r['included']]
    assert len({r['rgb_dimensions_sha256'] for r in eligible})==len(eligible)
    eligible_train=[r for r in eligible if r['official_split']=='trainval']
    query=sorted([r for r in eligible if r['official_split']=='test'],key=lambda r:(r['class_id'],r['image_id']))
    assert len(eligible_train)==49967 and len(query)==9990
    pools={}
    for c in range(100):
        candidates=[r for r in eligible_train if r['class_id']==c]
        pools[str(c)]=[r['image_id'] for r in sorted(candidates,key=lambda r:(key('candidate',c,r['image_id']),r['image_id']))[:100]]
        assert len(pools[str(c)])==100
    episodes=[]
    for e in range(12):
        ids=[];labels=[];ranks=[];folds=[]
        for c in range(100):
            ordered=sorted(pools[str(c)],key=lambda i:(key('episode',e,c,i),i))[:16]
            ids.extend(ordered);labels.extend([c]*16);ranks.extend(range(16));folds.extend(j%4 for j in range(16))
        episodes.append({'episode':e,'ids_16':ids,'labels_16':labels,'rank_16':ranks,'fold_16':folds})
    support_ids=sorted({i for e in episodes for i in e['ids_16']})
    qids={r['image_id'] for r in query}
    assert not set(support_ids)&qids
    assert not {byid[i]['rgb_dimensions_sha256'] for i in support_ids}&{r['rgb_dimensions_sha256'] for r in query}
    compact=lambda r:{'id':r['image_id'],'index':r['raw_array_index'],'label':r['class_id'],'rgb_hash':r['rgb_dimensions_sha256']}
    timing_ids=sorted(support_ids,key=lambda i:(key('support_only_timing',i),i))[:8]
    for i in timing_ids:
        r=byid[i];arr=raw[r['raw_array_index']]
        assert hashlib.sha256(struct.pack('>II',32,32)+arr.tobytes()).hexdigest()==r['rgb_dimensions_sha256']
    tests=[]
    for e in episodes:
        labels=np.array(e['labels_16']);rank=np.array(e['rank_16']);fold=np.array(e['fold_16'])
        for B in [8,16]:
            mask=rank<B
            assert collections.Counter(labels[mask])=={c:B for c in range(100)}
            for f in range(4):assert collections.Counter(labels[mask&(fold==f)])=={c:B//4 for c in range(100)}
            tests.append({'episode':e['episode'],'B':B,'originals':int(mask.sum()),'per_fold_originals':int((mask&(fold==0)).sum())})
    identity={'dataset':'CIFAR-100','fine_classes':100,'salt':SALT,'source_manifest_sha256':receipt['manifest_sha256'],
              'source_array_sha256':receipt['raw_uint8_npy_sha256'],'candidate_pool_ids':pools,
              'episodes':episodes,'support_union':[compact(byid[i]) for i in support_ids],
              'query':[compact(r) for r in query],'timing_support_ids':timing_ids,
              'query_rule':'all eligible official test images; min(100,n_class); historical duplicate/conflict exclusion only',
              'prior_data_exposure':'used by earlier local research; exploratory cross-dataset replication, not fresh confirmation'}
    dump('IDENTITIES.json',identity)
    qcounts=collections.Counter(r['class_id'] for r in query)
    originalcounts=collections.Counter((r['official_split'],r['class_id']) for r in records)
    assert all(originalcounts[('trainval',c)]==500 and originalcounts[('test',c)]==100 for c in range(100))
    models={}
    for name,rel,expected in [
        ('dinov2_small','research-shared-theta-ad-20261030/models/dinov2-small/model.safetensors','ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'),
        ('deit_tiny','research-view-route-20261030/assets/deit_tiny_patch16_224-a1311bcf.pth','a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97')]:
        p=PROJECT/rel;assert sha(p)==expected;models[name]={'path':rel,'bytes':p.stat().st_size,'sha256':expected}
    audit={'status':'PASS','seconds':time.perf_counter()-t,'dataset_locked':'all-100-class CIFAR-100','no_other_dataset_evaluated':True,
           'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,'npy_sha256':receipt['raw_uint8_npy_sha256'],
           'npy_bytes':(DATA/'data/original_rgb_uint8.npy').stat().st_size,'metadata_sha256':receipt['manifest_sha256'],
           'eligible_train':len(eligible_train),'eligible_query':len(query),'all_100_classes_retained':True,
           'candidate_pool_originals':sum(map(len,pools.values())),'support_union_originals':len(support_ids),
           'query_per_class_counts':dict(sorted(qcounts.items())),'minimum_query_per_class':min(qcounts.values()),
           'support_query_identity_and_rgb_overlap':0,'excluded_reasons':dict(collections.Counter(r['exclusion_reason'] for r in records if not r['included'])),
           'timing_support_ids':timing_ids,'identity_contract_checks':tests,'identities_sha256':sha(ROOT/'IDENTITIES.json'),'models':models,
           'new_encoder_forwards':0,'new_real_head_fits':0,'new_query_utility':0,'new_data_download_bytes':0,
           'pickle_execution':False,'raw_array_loaded_allow_pickle':False}
    dump('ASSET_AUDIT.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['query_per_class_counts','identity_contract_checks','models']},indent=2))

if __name__=='__main__':main()
