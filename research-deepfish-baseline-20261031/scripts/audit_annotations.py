"""Read published annotations unchanged; audit identities/geometry, never derive boxes."""
import collections,csv,hashlib,itertools,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
ANN=ROOT/'assets/annotation_docs/FISH/annotations'
archives=json.loads((ROOT/'reports/FISH_archive_members.json').read_text())
files={x['name']:x for x in archives if x['type']=='file'}
data={p.stem:json.loads(p.read_text()) for p in ANN.glob('*.json')}
summary={};identities=[]
for split,d in sorted(data.items()):
 images={x['id']:x for x in d['images']};counts=collections.Counter(x['image_id'] for x in d['annotations']); cats={x['id'] for x in d['categories']};bad=[]
 prefix='test' if split=='test' else 'train';missing=[]
 for im in d['images']:
  fn=pathlib.PurePosixPath(im['file_name'])
  if fn.is_absolute() or '..' in fn.parts: raise ValueError('Unsafe annotation filename')
  name=f'FISH/{prefix}/{fn}'
  if name not in files:missing.append(name)
  if 'shot' in split:
   own=[a for a in d['annotations'] if a['image_id']==im['id']]
   full=[a for a in data['train']['annotations'] if a['image_id']==im['id']]
   train_im=next((x for x in data['train']['images'] if x['id']==im['id']),None)
   identities.append({'split':split,'image_id':im['id'],'file_name':im['file_name'],'width':im['width'],'height':im['height'],'archive_member':name,'image_sha256':files.get(name,{}).get('sha256'),'support_annotations':own,'full_train_annotation_count_for_id':len(full),'omitted_full_train_annotations':len(full)-len(own),'train_image_exact_match':train_im==im,'support_annotations_exact_train_subset':all(a in full for a in own)})
 for a in d['annotations']:
  im=images.get(a['image_id']);box=a.get('bbox');problems=[]
  if im is None:problems.append('dangling_image')
  if a['category_id'] not in cats:problems.append('unknown_category')
  if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):problems.append('invalid_bbox')
  else:
   x,y,w,h=box
   if w<=0 or h<=0:problems.append('nonpositive_bbox')
   if im and (x<0 or y<0 or x+w>im['width'] or y+h>im['height']):problems.append('outside_image')
   if not math.isclose(a.get('area',float('nan')),w*h,rel_tol=1e-7,abs_tol=1e-6):problems.append('area_not_box_area')
  if problems:bad.append({'annotation_id':a['id'],'image_id':a['image_id'],'issues':problems})
 imids=[i['id'] for i in d['images']];annids=[i['id'] for i in d['annotations']];names=[i['file_name'] for i in d['images']]
 summary[split]={'images':len(imids),'annotations':len(annids),'categories':d['categories'],'annotation_file_sha256':hashlib.sha256((ANN/(split+'.json')).read_bytes()).hexdigest(),'duplicate_image_ids':len(imids)-len(set(imids)),'duplicate_annotation_ids':len(annids)-len(set(annids)),'duplicate_filenames':len(names)-len(set(names)),'missing_image_files':missing,'empty_images':sum(counts[i]==0 for i in imids),'annotation_count_histogram':dict(sorted(collections.Counter(counts[i] for i in imids).items())),'geometry_issues':bad,'image_dimensions':dict(collections.Counter(str((i['width'],i['height'])) for i in d['images']))}

def split_names(s):return {i['file_name'] for i in data[s]['images']}
def split_sha(s):return {files['FISH/'+('test' if s=='test' else 'train')+'/'+i['file_name']]['sha256'] for i in data[s]['images']}
def groups(s):return {re.sub(r'_f\d+\.[^.]+$','',i['file_name']) for i in data[s]['images']}
pairs={}
for a,b in itertools.combinations(sorted(data),2):
 pairs[a+'__'+b]={'filename_overlap':sorted(split_names(a)&split_names(b)),'content_sha256_overlap':sorted(split_sha(a)&split_sha(b)),'numeric_image_id_overlap_count':len({i['id'] for i in data[a]['images']}&{i['id'] for i in data[b]['images']}),'filename_prefix_group_overlap_count':len(groups(a)&groups(b))}
report={'splits':summary,'pairs':pairs,'support_identities':identities,'train_test_filename_group_heuristic':{'rule':'remove trailing _f<digits>.<extension>; this is only a filename heuristic, not a verified site/video mapping','train_groups':len(groups('train')),'test_groups':len(groups('test')),'shared_groups':sorted(groups('train')&groups('test'))},'dev_split_present':False,'archive_annotation_names':sorted(str(p.relative_to(ROOT)) for p in ANN.glob('*.json')),'label_creation_or_conversion_metadata_present':False,'license_file_in_archive':False,'boxes_created_by_this_audit':0}
(ROOT/'reports/ANNOTATION_AUDIT.json').write_text(json.dumps(report,indent=2))
with (ROOT/'reports/SUPPORT_IDENTITIES.csv').open('w',newline='') as f:
 fields=['split','image_id','file_name','image_sha256','annotation_ids','bboxes','full_train_annotation_count','omitted_full_train_annotations','matches_train']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in identities:w.writerow({'split':r['split'],'image_id':r['image_id'],'file_name':r['file_name'],'image_sha256':r['image_sha256'],'annotation_ids':json.dumps([a['id'] for a in r['support_annotations']]),'bboxes':json.dumps([a['bbox'] for a in r['support_annotations']]),'full_train_annotation_count':r['full_train_annotation_count_for_id'],'omitted_full_train_annotations':r['omitted_full_train_annotations'],'matches_train':r['train_image_exact_match'] and r['support_annotations_exact_train_subset']})
print(json.dumps({'splits':{k:{a:v for a,v in d.items() if a not in ['annotation_count_histogram','geometry_issues']}|{'geometry_issue_count':len(d['geometry_issues'])} for k,d in summary.items()},'train_test':{k:len(v) if isinstance(v,list) else v for k,v in pairs['test__train'].items()},'support':[{k:v for k,v in r.items() if k not in ['support_annotations','image_sha256']} for r in identities],'group_overlap':len(groups('train')&groups('test'))},indent=2))
