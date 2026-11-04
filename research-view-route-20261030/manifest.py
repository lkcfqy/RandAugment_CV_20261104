import hashlib,json,struct,time
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    start=time.time();annotations={}
    for split in ['trainval','test']:
        for line in (ROOT/'data'/'annotations'/f'{split}.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'):continue
            ident,cls,species,breed=line.split()
            assert ident not in annotations
            annotations[ident]={'official_split':split,'class_id':int(cls),'species':int(species),'breed_id':int(breed)}
    records=[]
    for p in sorted((ROOT/'data'/'images').glob('*.jpg')):
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            w,h=im.size;pixel=hashlib.sha256(struct.pack('>II',w,h)+im.tobytes()).hexdigest()
        records.append({'image_id':p.stem,'path':str(p.relative_to(ROOT)),**annotations.get(p.stem,{}),
                        'width':w,'height':h,'file_sha256':sha(p),'rgb_dimensions_sha256':pixel})
    ids={r['image_id'] for r in records};assert set(annotations)<=ids
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    for key in ['file_sha256','rgb_dimensions_sha256']:
        seen={}
        for i,r in enumerate(records):
            if r[key] in seen:union(i,seen[r[key]])
            else:seen[r[key]]=i
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[find(i)].append(r)
    duplicate_groups=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:r['image_id'])
        labels={r['class_id'] for r in group if 'class_id' in r}
        relevant=[r for r in group if 'class_id' in r]
        tests=[r for r in relevant if r['official_split']=='test']
        keep=None if len(labels)>1 or not relevant else (tests or relevant)[0]['image_id']
        for r in group:
            r['included']=r['image_id']==keep
            r['exclusion_reason']=None if r['included'] else ('label_conflict' if len(labels)>1 else 'not_in_official_split' if 'class_id' not in r else 'duplicate')
            r['duplicate_group_representative']=group[0]['image_id']
        if len(group)>1 or len(labels)>1:duplicate_groups.append({'ids':[r['image_id'] for r in group],'labels':sorted(labels),'kept':keep})
    byid={r['image_id']:r for r in records}
    validation=[];supports={str(s):[] for s in [11,22,33]}
    for cls in range(1,38):
        pool=[r['image_id'] for r in records if r['included'] and r.get('official_split')=='trainval' and r['class_id']==cls]
        def order(domain,seed=None):
            prefix=f'pets-route-v1|{domain}|'+(f'{seed}|' if seed is not None else '')+f'{cls}|'
            return lambda ident:(hashlib.sha256((prefix+ident).encode()).hexdigest(),ident)
        val=sorted(pool,key=order('validation'))[:16]
        assert len(val)==16,('insufficient validation',cls)
        validation.extend(val);remaining=set(pool)-set(val)
        for s in supports:
            chosen=sorted(remaining,key=order('support',s))[:16]
            assert len(chosen)==16,('insufficient support',cls)
            supports[s].extend(chosen)
    test=sorted(r['image_id'] for r in records if r['included'] and r.get('official_split')=='test')
    for s,support in supports.items():
        assert not set(support)&set(validation) and not set(support)&set(test)
        assert Counter(byid[x]['class_id'] for x in support)=={c:16 for c in range(1,38)}
    splits={'version':'pets-route-v1','class_id_encoding':'official 1..37; training labels class_id-1',
            'validation':validation,'support':supports,'test':test,
            'rgb_hash_encoding':'SHA256(big-endian uint32 width + uint32 height + EXIF-transposed RGB bytes)',
            'extra_validation_labels_per_class':16,'test_read_only_for_hash_before_selection':True}
    dump(ROOT/'data-manifest.json',records);dump(ROOT/'splits.json',splits);dump(ROOT/'duplicates.json',duplicate_groups)
    summary={'status':'passed','seconds':time.time()-start,'total_image_files':len(records),'official_rows':len(annotations),
             'official_counts':dict(Counter(v['official_split'] for v in annotations.values())),
             'included_counts':dict(Counter(r.get('official_split') for r in records if r['included'])),
             'exclusion_counts':dict(Counter(r['exclusion_reason'] for r in records if not r['included'])),
             'duplicate_groups':len(duplicate_groups),'validation_count':len(validation),'support_per_seed':{s:len(v) for s,v in supports.items()},
             'test_count':len(test),'manifest_sha256':sha(ROOT/'data-manifest.json'),'splits_sha256':sha(ROOT/'splits.json')}
    dump(ROOT/'data-validation.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
