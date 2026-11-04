"""Freeze metadata-only IDs, then decode only those chosen images using existing CPU tools."""
from contracts import *
import csv,collections,copy,sys

def select():
    assert not (P/'SELECTION.json').exists(),'Refuse resampling'
    full=list(csv.DictReader((ASSET/'reports/ALL_FRAME_QUALIFICATION.csv').open()))
    allnames=sorted(r['filename'] for r in full)
    ids={n:i+1 for i,n in enumerate(allnames)}
    pools={k:list(csv.DictReader((ASSET/f'reports/PROPOSED_{v}_POOL_FRAMES.csv').open())) for k,v in [('support','SUPPORT'),('dev','DEVELOPMENT'),('confirm','FUTURE_CONFIRMATION')]}
    vs=sorted({r['video'] for r in pools['support'] if r['one_total_box_candidate']=='True'},key=video_sort_key)
    assert len(vs)==7
    per={v:sorted([r for r in pools['support'] if r['video']==v and r['one_target_candidate']=='True'],key=lambda x:int(x['frame'])) for v in vs}
    assert all(len(per[v])>=2 for v in vs[2:5])
    groups={}
    for episode in (0,1):
        chosen=vs[:5] if episode==0 else vs[2:]
        groups['support'+str(episode)]=[]
        for v in chosen:
            j=support_index(len(per[v]),v in vs[2:5],episode)
            row=copy.deepcopy(per[v][j]);row.update(candidate_n=len(per[v]),selected_index=j,rank=vs.index(v))
            groups['support'+str(episode)].append(row)
    for split in ('dev','confirm'):
        groups[split]=[]
        for v in sorted({r['video'] for r in pools[split]}):
            rows=sorted([r for r in pools[split] if r['video']==v],key=lambda x:int(x['frame']))
            for j in cap_indices(len(rows)):
                row=copy.deepcopy(rows[j]);row.update(candidate_n=len(rows),selected_index=j)
                groups[split].append(row)
    for name,rows in groups.items():
        for row in rows:
            assert covered(row['video'],int(row['frame'])) and row['eligible_frame_metadata']=='True'
            row['image_id']=ids[row['filename']]
    names=[r['filename'] for rows in groups.values() for r in rows]
    assert len(names)==len(set(names))==418
    for name in ('support0','support1'):
        assert len(groups[name])==5 and sum(int(r['fish_target_boxes']) for r in groups[name])==5
    dates={k:sorted({r['date'] for r in pools[k]}) for k in pools}
    assert all(not(set(dates[a])&set(dates[b])) for a,b in [('support','dev'),('support','confirm'),('dev','confirm')])
    out={'status':'FROZEN_BEFORE_PIXEL_DECODE_MODEL_OR_EFFECTS','UTC':utc(),'salt':SALT,'salt_encoding':'UTF-8 of salt + vertical-bar + video_ID','video_order':vs,
         'video_candidate_counts':{v:len(per[v]) for v in vs},'support_video_eligibility':'Seven videos from prior strict one-total-fish-box qualification; within them candidate frames retain exactly one fish target, with any other animal annotation retained in provenance.',
         'shared_videos':vs[2:5],'independent_replicates':False,'same_fish_identity_excluded':False,
         'dates':dates,'groups':groups,'counts':{k:len(v) for k,v in groups.items()},
         'target_box_counts':{k:sum(int(r['fish_target_boxes']) for r in v) for k,v in groups.items()},
         'zero_target_frames':{k:sum(int(r['fish_target_boxes'])==0 for r in v) for k,v in groups.items()},
         'decode':'Existing OpenCV CAP_FFMPEG CPU; sequential grab, retrieve selected n at zero-based index n-1; INTER_CUBIC from1920x1080 to960x540; losslessPNG; not claimed pixel-identical to official ffmpeg CLI',
         'source_pool_sha256':{k:sha(ASSET/f'reports/PROPOSED_{v}_POOL_FRAMES.csv') for k,v in [('support','SUPPORT'),('dev','DEVELOPMENT'),('confirm','FUTURE_CONFIRMATION')]}}
    atomic(P/'SELECTION.json',out)
    print(json.dumps({k:out[k] for k in ['counts','target_box_counts','zero_target_frames','video_order','video_candidate_counts']}),flush=True)

def decode():
    import cv2,numpy as np
    cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
    frozen=read(P/'SELECTION.json');assert not (P/'DATA_READY.json').exists()
    guard=Guard();guard.check(extra=500*2**20)
    rows=[r for group in frozen['groups'].values() for r in group]
    byvideo=collections.defaultdict(dict)
    for r in rows:byvideo[r['video']][int(r['frame'])]=r
    videos={r['video']:r for r in csv.DictReader((ASSET/'reports/VIDEO_COVERAGE.csv').open())}
    ann=collections.defaultdict(list)
    for s in ('train','valid','test'):
        co=read(ASSET/f'original_v4/annotations/annotations_COCO/{s}_groundtruth.json');im={i['id']:i for i in co['images']}
        for a in co['annotations']:ann[im[a['image_id']]['file_name']].append({**a,'original_split':s})
    (P/'data/images').mkdir(parents=True);(P/'data/annotations').mkdir()
    receipts={};t0=time.monotonic()
    for v,selected in byvideo.items():
        guard.check();source=ASSET/'original_v4'/videos[v]['archive_path'];assert sha(source)==videos[v]['sha256']
        cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_NONE,cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened()
        for n in range(1,max(selected)+1):
            assert cap.grab(),(v,n)
            if n not in selected:continue
            ok,native=cap.retrieve();assert ok and native.shape==(1080,1920,3) and native.dtype==np.uint8
            assert cap.get(cv2.CAP_PROP_POS_FRAMES)==n
            row=selected[n];im=cv2.resize(native,(960,540),interpolation=cv2.INTER_CUBIC);dest=P/'data/images'/row['filename']
            assert cv2.imwrite(str(dest),im)
            receipts[row['filename']]={'image_id':row['image_id'],'video':v,'frame_number':n,'source_video_sha256':videos[v]['sha256'],'image_sha256':sha(dest),'bytes':dest.stat().st_size,'native_shape':[1080,1920,3],'shape':[540,960,3],'decoded_BGR_sha256':hashlib.sha256(native.tobytes()).hexdigest()}
        cap.release()
    provenance={}
    for group,items in frozen['groups'].items():
        dataset={'info':{'description':'Brackish v4 fixed same-camera date-disjoint pilot; covered-no-target is not proven visual absence'},'licenses':[{'id':1,'name':'CC BY-SA 4.0; original COCO MIT discrepancy unresolved','url':'https://creativecommons.org/licenses/by-sa/4.0/'}],'images':[],'annotations':[],'categories':[{'id':1,'name':'fish','supercategory':'animal'}]}
        for row in items:
            iid=row['image_id'];name=row['filename'];original=ann[name]
            provenance[name]={'image_id':iid,'source_COCO_image_ids':row['COCO_split_image_ids'],'original_annotations':original,'coverage':row['coverage'],'original_splits':row['published_splits'],'fish_map':[1,2],'non_target_categories':[3,4,5,6]}
            dataset['images'].append({'id':iid,'file_name':name,'width':960,'height':540,'license':1})
            for a in mapped_fish(original,iid):
                a['id']=len(dataset['annotations'])+1;dataset['annotations'].append(a)
        assert len(dataset['annotations'])==frozen['target_box_counts'][group]
        atomic(P/f'data/annotations/{group}.json',dataset)
    atomic(P/'SOURCE_ANNOTATION_PROVENANCE.json',provenance)
    atomic(P/'IMAGE_MANIFEST.json',receipts)
    ready={'status':'COMPLETE','images':len(receipts),'image_bytes':sum(r['bytes'] for r in receipts.values()),'decode_save_seconds':time.monotonic()-t0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'selection_sha256':sha(P/'SELECTION.json'),'annotations':{k:sha(P/f'data/annotations/{k}.json') for k in frozen['groups']},'image_manifest_sha256':sha(P/'IMAGE_MANIFEST.json'),'new_installs_downloads_uploads_CU':0}
    atomic(P/'DATA_READY.json',ready);print(json.dumps(ready),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='select':select()
    elif sys.argv[1]=='decode':decode()
