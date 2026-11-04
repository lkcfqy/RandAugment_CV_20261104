"""Inspect published annotations/filenames only. Never imports release scripts or decodes videos."""
import pathlib,json,csv,collections,re,math,hashlib,itertools,datetime
P=pathlib.Path(__file__).resolve().parent;D=P/'original_v4';O=P/'reports'
def dump(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def table(name,rows):
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
video_re=re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})to(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$')
parse=lambda x:datetime.datetime.strptime(x,'%Y-%m-%d_%H-%M-%S')
videos={}
manifest={x['path']:x for x in json.loads((O/'EXTRACTED_FILE_SHA256.json').read_text())}
for f in sorted((D/'dataset/videos').rglob('*.avi')):
    match=video_re.fullmatch(f.stem);assert match,f.name
    start,end,suffix=match.groups();a,b=parse(start),parse(end);assert a<=b
    assert f.stem not in videos
    videos[f.stem]={'video':f.stem,'archive_path':str(f.relative_to(D)),'start_filename':start,'end_filename':end,'start_date':a.date().isoformat(),'end_date':b.date().isoformat(),'encoded_duration_seconds':(b-a).total_seconds(),'suffix':suffix,'folder_predominant_label_only':f.parent.name,'bytes':f.stat().st_size,'sha256':manifest[str(f.relative_to(D))]['sha256']}
assert len(videos)==89
partial={'2019-03-19_17-07-53to2019-03-19_17-08-34_1':200,'2019-03-19_18-01-56to2019-03-19_18-02-13_1':100}
split_lists={s:(D/(s+'.txt')).read_text().splitlines() for s in ['train','valid','test']}
file_to_splits=collections.defaultdict(list)
for s,names in split_lists.items():
    assert len(names)==len(set(names))
    for n in names:file_to_splits[n].append(s)
frames={};raw_yolo={};parse_errors=[]
for f in sorted((D/'annotations/annotations_YOLO').glob('*.txt')):
    video,n=f.stem.rsplit('-',1);assert video in videos and n.isdigit()
    number=int(n);fname=f.stem+'.png';assert fname not in frames
    lines=[x.split() for x in f.read_text().splitlines() if x.strip()]
    try:ys=[tuple(float(t) for t in x) for x in lines];assert all(len(x)==5 and all(math.isfinite(v) for v in x) for x in ys)
    except Exception:parse_errors.append(str(f));ys=[]
    raw_yolo[fname]=ys
    coverage='outside_declared_annotation_coverage' if video in partial and number>partial[video] else 'within_declared_annotation_coverage'
    frames[fname]={'filename':fname,'video':video,'date':videos[video]['start_date'],'frame':number,'published_splits':';'.join(file_to_splits.get(fname,[])),'coverage':coverage,'YOLO_rows':len(ys),'COCO_rows':0,'fish_target_boxes':0,'other_animal_boxes':0,'COCO_split_image_ids':'','one_target_candidate':False,'one_total_box_candidate':False}
assert not parse_errors
assert set(file_to_splits)<=set(frames)
coco_per_frame=collections.defaultdict(list);aau_per_frame=collections.defaultdict(list);split_stats={};categories=None;invalid=[]
for s in split_lists:
    co=json.loads((D/f'annotations/annotations_COCO/{s}_groundtruth.json').read_text());cats={x['id']:x['name'] for x in co['categories']}
    if categories is None:categories=cats
    assert cats==categories
    images={x['id']:x for x in co['images']};assert len(images)==len(co['images'])
    names=[x['file_name'] for x in co['images']];assert len(names)==len(set(names)) and set(names)<=set(split_lists[s])
    assert len({a['id'] for a in co['annotations']})==len(co['annotations'])
    for a in co['annotations']:
        im=images[a['image_id']];name=im['file_name'];x,y,w,h=a['bbox'];cat=cats[a['category_id']]
        if not all(math.isfinite(float(v)) for v in [x,y,w,h,a['area']]) or w<=0 or h<=0 or x<0 or y<0 or x+w>im['width'] or y+h>im['height'] or a['area']!=w*h or a.get('iscrowd',0) or a.get('ignore',0):invalid.append({'split':s,'annotation_id':a['id'],'filename':name,'annotation':a})
        coco_per_frame[name].append((a['category_id'],x,y,w,h))
    for im in co['images']:
        f=frames[im['file_name']];f['COCO_split_image_ids']+=f"{s}:{im['id']};"
        assert (im['width'],im['height'])==(960,540)
    with (D/f'annotations/annotations_AAU/{s}.csv').open() as stream:
        for a in csv.DictReader(stream,delimiter=';'):
            cat=next(k for k,v in cats.items() if v==a['Annotation tag']);x,y,u,v=[float(a[k]) for k in ['Upper left corner X','Upper left corner Y','Lower right corner X','Lower right corner Y']]
            aau_per_frame[a['Filename']].append((cat,x,y,u-x,v-y))
    split_stats[s]={'frame_list_count':len(split_lists[s]),'COCO_images':len(co['images']),'COCO_annotations':len(co['annotations']),'COCO_licenses_field':co['licenses'],'category_counts':dict(collections.Counter(cats[a['category_id']] for a in co['annotations'])),'COCO_omitted_frame_list_entries':len(set(split_lists[s])-set(names)),'videos':len({frames[n]['video'] for n in split_lists[s]}),'dates':len({frames[n]['date'] for n in split_lists[s]})}
assert categories=={1:'fish',2:'small_fish',3:'crab',4:'shrimp',5:'jellyfish',6:'starfish'}
mismatch_aau=[];mismatch_yolo=[];out_coverage=[]
for name,fr in frames.items():
    co=coco_per_frame[name];aa=aau_per_frame[name]
    if collections.Counter(co)!=collections.Counter(aa):mismatch_aau.append(name)
    expected=[(float(c),float(int(x+w/2)),float(int(y+h/2)),float(int(w)),float(int(h))) for c,x,y,w,h in co]
    if collections.Counter(expected)!=collections.Counter(raw_yolo[name]):mismatch_yolo.append(name)
    fr['COCO_rows']=len(co);fr['fish_target_boxes']=sum(c in (1,2) for c,*_ in co);fr['other_animal_boxes']=len(co)-fr['fish_target_boxes']
    if fr['coverage']=='outside_declared_annotation_coverage':out_coverage.append(fr)
    eligible=fr['coverage']=='within_declared_annotation_coverage' and bool(fr['published_splits']) and name not in mismatch_aau
    fr['eligible_frame_metadata']=eligible
    fr['one_target_candidate']=eligible and fr['fish_target_boxes']==1
    fr['one_total_box_candidate']=eligible and len(co)==1 and fr['fish_target_boxes']==1
all_rows=sorted(frames.values(),key=lambda x:(x['date'],x['video'],x['frame']))
table('ALL_FRAME_QUALIFICATION.csv',all_rows)
table('UNCOVERED_TAIL_FRAMES.csv',sorted(out_coverage,key=lambda x:(x['video'],x['frame'])))
video_rows=[]
for name,meta in videos.items():
    fs=[x for x in all_rows if x['video']==name];numbers=[x['frame'] for x in fs];eligible=[x for x in fs if x['eligible_frame_metadata']]
    counts=collections.Counter(categories[c] for f in fs for c,*_ in coco_per_frame[f['filename']])
    row={**meta,'listed_frame_count':len(fs),'frame_min':min(numbers),'frame_max':max(numbers),'missing_between_frame_min_max':len(set(range(min(numbers),max(numbers)+1))-set(numbers)),'declared_annotation_limit':partial.get(name,'all frames per publisher'),'outside_coverage_listed_frames':sum(x['coverage']=='outside_declared_annotation_coverage' for x in fs),'outside_coverage_in_official_split':sum(x['coverage']=='outside_declared_annotation_coverage' and bool(x['published_splits']) for x in fs),'eligible_frames':len(eligible),'all_animal_boxes':sum(x['COCO_rows'] for x in eligible),'fish_target_boxes':sum(x['fish_target_boxes'] for x in eligible),'no_animal_annotation_frames':sum(x['COCO_rows']==0 for x in eligible),'zero_target_frames':sum(x['fish_target_boxes']==0 for x in eligible),'one_target_frames':sum(x['one_target_candidate'] for x in fs),'one_total_box_frames':sum(x['one_total_box_candidate'] for x in fs),**{s+'_frames':sum(s in x['published_splits'].split(';') for x in fs) for s in split_lists},**{'class_'+k:counts.get(k,0) for k in categories.values()}}
    video_rows.append(row)
table('VIDEO_COVERAGE.csv',video_rows)
date_rows=[]
for date in sorted({x['date'] for x in all_rows}):
    fs=[x for x in all_rows if x['date']==date];eligible=[x for x in fs if x['eligible_frame_metadata']]
    date_rows.append({'date':date,'videos':len({x['video'] for x in fs}),'eligible_frames':len(eligible),'fish_target_boxes':sum(x['fish_target_boxes'] for x in eligible),'other_animal_boxes':sum(x['other_animal_boxes'] for x in eligible),'no_animal_annotation_frames':sum(x['COCO_rows']==0 for x in eligible),'zero_target_frames':sum(x['fish_target_boxes']==0 for x in eligible),'one_target_frames':sum(x['one_target_candidate'] for x in eligible),'one_target_videos':len({x['video'] for x in eligible if x['one_target_candidate']}),'one_total_box_frames':sum(x['one_total_box_candidate'] for x in eligible),'one_total_box_videos':len({x['video'] for x in eligible if x['one_total_box_candidate']}),'original_split_names':';'.join(sorted(set(s for x in fs for s in x['published_splits'].split(';') if s)))})
table('DATE_INVENTORY.csv',date_rows)
overlap_intervals=[]
for a,b in itertools.combinations(video_rows,2):
    begin=max(parse(a['start_filename']),parse(b['start_filename']));end=min(parse(a['end_filename']),parse(b['end_filename']))
    if begin<end:overlap_intervals.append({'video_a':a['video'],'video_b':b['video'],'filename_interval_overlap_seconds':(end-begin).total_seconds(),'actual_pixel_overlap_not_checked':True})
sets={s:set(v) for s,v in split_lists.items()};set_overlap={a+'__'+b:len(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)}
split_leak={'split_counts':split_stats,'duplicate_frame_names_between_splits':set_overlap,'all3_split_video_count':sum(all(x[s+'_frames']>0 for s in split_lists) for x in video_rows),'all3_split_date_count':sum(x['original_split_names']=='test;train;valid' for x in date_rows),'within_video_adjacent_frames_in_different_split':sum(bool(a['published_splits'] and b['published_splits'] and a['video']==b['video'] and b['frame']-a['frame']==1 and a['published_splits']!=b['published_splits']) for a,b in zip(all_rows,all_rows[1:])),'filename_interval_overlaps':overlap_intervals,'cross_midnight_videos':[x['video'] for x in video_rows if x['start_date']!=x['end_date']]}
dump('ORIGINAL_SPLIT_AUDIT.json',split_leak)
qualified=[x for x in all_rows if x['eligible_frame_metadata']]
dates=sorted({x['date'] for x in qualified});assert len(dates)>=3
n=len(dates);a=max(1,math.floor(.6*n));b=max(a+1,math.floor(.8*n));b=min(b,n-1)
blocks={'support_pool':dates[:a],'development_pool':dates[a:b],'future_confirmation_pool':dates[b:]}
block_stats={}
for block,ds in blocks.items():
    fs=[x for x in qualified if x['date'] in ds]
    block_stats[block]={'dates':ds,'videos':len({x['video'] for x in fs}),'frames':len(fs),'fish_target_boxes':sum(x['fish_target_boxes'] for x in fs),'other_animal_boxes':sum(x['other_animal_boxes'] for x in fs),'no_animal_annotation_frames':sum(x['COCO_rows']==0 for x in fs),'zero_target_frames':sum(x['fish_target_boxes']==0 for x in fs),'one_target_frames':sum(x['one_target_candidate'] for x in fs),'one_target_videos':len({x['video'] for x in fs if x['one_target_candidate']}),'one_total_box_frames':sum(x['one_total_box_candidate'] for x in fs),'one_total_box_videos':len({x['video'] for x in fs if x['one_total_box_candidate']})}
    table('PROPOSED_'+block.upper()+'_FRAMES.csv',fs)
proposal={'status':'metadata feasibility proposal only; no experiment/episode selected or authorized','rule':'all distinct encoded acquisition dates sorted chronologically, first floor(.6D) support pool, next to floor(.8D) dev, rest future confirmation; whole dates and clips retained, not original frame-random split','blocks':block_stats,'five_distinct_videos_one_target_each_feasible':block_stats['support_pool']['one_target_videos']>=5,'five_distinct_videos_exactly_one_total_box_each_feasible':block_stats['support_pool']['one_total_box_videos']>=5,'all_target_boxes_retained_per_image':True,'other_animals_definition':'non-target background for binary fish task, not ignored to hide false positives; original labels untouched','single_class_map':{'fish':1,'small_fish':1,'crab':'non-target','shrimp':'non-target','jellyfish':'non-target','starfish':'non-target'},'no_effect_or_pixel_selection':True,'dates_do_not_prove_independent_sites_or_camera_sessions':True,'date_timezone_clock_verification':'not provided/independently verified','partial_tail_exclusion_required':partial,'cross_midnight_status':'must keep intersecting dates together if found; none expected; see original split audit'}
dump('DATE_BLOCK_AND_5SHOT_FEASIBILITY.json',proposal)
quality={'videos':len(videos),'encoded_dates':dates,'YOLO_frame_files':len(frames),'original_split_frames':len(file_to_splits),'COCO_annotated_images':len([x for x in frames.values() if x['COCO_rows']]),'all_original_COCO_annotations':sum(x['COCO_rows'] for x in frames.values()),'category_counts':dict(collections.Counter(categories[c] for anns in coco_per_frame.values() for c,*_ in anns)),'AAU_COCO_mismatch_frames':mismatch_aau,'YOLO_published_integer_pixel_center_one_based_conversion_mismatch_frames':mismatch_yolo,'YOLO_class_values':sorted({int(a[0]) for ys in raw_yolo.values() for a in ys}),'YOLO_max_value_after_class':max(max(x[1:]) for ys in raw_yolo.values() for x in ys),'YOLO_rows':sum(len(v) for v in raw_yolo.values()),'out_of_bounds_original_box_records_retained':invalid,'outside_declared_coverage_frames':len(out_coverage),'outside_coverage_in_original_splits':{s:sum(s in x['published_splits'].split(';') for x in out_coverage) for s in split_lists},'qualified_frames_after_coverage_check':len(qualified),'qualified_no_animal_annotation_frames':sum(x['COCO_rows']==0 for x in qualified),'qualified_zero_target_frames':sum(x['fish_target_boxes']==0 for x in qualified),'uncovered_tail_nonempty_annotations':sum(x['COCO_rows']+x['YOLO_rows'] for x in out_coverage),'COCO_omitted_covered_zero_annotation_frames':sum(x['COCO_rows']==0 for x in qualified),'publisher_license':'CC BY-SA4.0','embedded_COCO_license':'MIT','conversion_source_hardcodes_MIT':True,'license_conflict_resolved_by_author':False,'original_annotation_files_modified':False,'upstream_code_executed':False,'pixels_decoded':False,'model_calls':0}
dump('DATA_QUALITY_AND_MAPPING.json',quality)
print(json.dumps({'quality':{k:v for k,v in quality.items() if k not in ['out_of_bounds_original_box_records_retained','AAU_COCO_mismatch_frames','YOLO_published_integer_pixel_center_one_based_conversion_mismatch_frames']},'AAU_COCO_mismatch_count':len(mismatch_aau),'YOLO_conversion_mismatch_count':len(mismatch_yolo),'out_of_bounds_original_box_count_retained':len(invalid),'date_rows':date_rows,'blocks':block_stats,'all3split_videos':split_leak['all3_split_video_count'],'all3split_dates':split_leak['all3_split_date_count'],'interval_overlaps':overlap_intervals},ensure_ascii=False))
