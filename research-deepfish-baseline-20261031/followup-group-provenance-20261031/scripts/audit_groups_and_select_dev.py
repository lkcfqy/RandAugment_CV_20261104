"""Read published train/support metadata only. Freeze a prefix-disjoint dev manifest, not new labels."""
import collections,csv,hashlib,json,math,pathlib,re,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
ASSET=pathlib.Path(sys.argv[1])
ANN=ASSET/'assets/annotation_docs/FISH/annotations'
def read(split):return json.loads((ANN/(split+'.json')).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def prefix(image):
 match=re.fullmatch(r'(.+)_f(\d+)\.jpg',image['file_name'])
 if not match:raise ValueError(image['file_name'])
 return match.group(1)
def frame(image):return int(re.search(r'_f(\d+)\.jpg$',image['file_name']).group(1))
def habitat(image):return image['file_name'].split('_')[0]
train=read('train');supports={k:read(str(k)+'_shot') for k in (1,5,10)}
byid={i['id']:i for i in train['images']};boxes=collections.defaultdict(list)
for a in train['annotations']:boxes[a['image_id']].append(a)
groups=collections.defaultdict(list)
for i in train['images']:groups[prefix(i)].append(i)
members={m['name']:m for m in json.loads((ASSET/'reports/FISH_archive_members.json').read_text())}
def count(images):return {'images':len(images),'boxes':sum(len(boxes[i['id']]) for i in images),'prefix_groups':len({prefix(i) for i in images}),'habitat_codes':len({habitat(i) for i in images})}
stats={'official_train':count(train['images']),'test_reopened':False,'definitions':{'prefix':'filename without terminal _f<digits>.jpg','habitat_code':'numeric first token; official slice_df_reg uses this as habitat selector','prefix_semantics':'not established as a unique video/site'},'shots':{}}
for k,split in supports.items():
 ps=sorted({prefix(i) for i in split['images']});hs=sorted({habitat(i) for i in split['images']})
 stats['shots'][str(k)]={'support_ids':sorted(i['id'] for i in split['images']),'support_prefixes':ps,'support_prefix_count':len(ps),'support_habitat_codes':hs,'support_habitat_count':len(hs),'remaining_train_excluding_support_prefixes':count([i for i in train['images'] if prefix(i) not in ps]),'remaining_train_excluding_support_habitats':count([i for i in train['images'] if habitat(i) not in hs])}
allhs={habitat(i) for s in supports.values() for i in s['images']}
stats['all_shots_union_habitat_exclusion']={'excluded_codes':sorted(allhs),**count([i for i in train['images'] if habitat(i) not in allhs])}
with (ROOT/'reports/TRAIN_GROUP_INVENTORY.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['prefix','habitat_code','train_images','train_boxes','minimum_frame','maximum_frame','support_1shot','support_5shot','support_10shot'])
 for p,images in sorted(groups.items()):w.writerow([p,habitat(images[0]),len(images),sum(len(boxes[i['id']]) for i in images),min(map(frame,images)),max(map(frame,images)),*[sum(prefix(i)==p for i in supports[k]['images']) for k in (1,5,10)]])
(ROOT/'reports/GROUP_AUDIT.json').write_text(json.dumps(stats,indent=2))
namespace='deepfish-prefix-dev-v1\n';excluded=set(stats['shots']['5']['support_prefixes'])
eligible=sorted((p for p in groups if p not in excluded),key=lambda p:(hashlib.sha256((namespace+p).encode()).hexdigest(),p))
chosen_groups=eligible[:8];selected=[];group_rows=[]
for p in chosen_groups:
 images=sorted(groups[p],key=lambda i:(frame(i),i['id']));n=min(8,len(images));lo,hi=frame(images[0]),frame(images[-1]);remaining=images.copy();chosen=[]
 for j in range(n):
  den=max(1,n-1);num=lo*den+j*(hi-lo)
  pick=min(remaining,key=lambda i:(abs(frame(i)*den-num),frame(i),i['id']));remaining.remove(pick);chosen.append(pick)
 chosen.sort(key=lambda i:(frame(i),i['id']));selected+=chosen
 group_rows.append({'prefix':p,'habitat_code':habitat(chosen[0]),'selection_sha256':hashlib.sha256((namespace+p).encode()).hexdigest(),'eligible_train_images':len(images),'selected_image_ids':[i['id'] for i in chosen],'selected_frames':[frame(i) for i in chosen],'selected_boxes':sum(len(boxes[i['id']]) for i in chosen)})
def record(i):
 member='FISH/train/'+i['file_name'];m=members[member]
 return {'image_id':i['id'],'file_name':i['file_name'],'prefix':prefix(i),'habitat_code':habitat(i),'frame':frame(i),'width':i['width'],'height':i['height'],'annotation_ids':sorted(a['id'] for a in boxes[i['id']]),'box_count':len(boxes[i['id']]),'archive_member':member,'image_bytes':m['size'],'image_sha256':m['sha256']}
dev={'status':'frozen metadata selection; no new annotations or official split edits','selection_version':'deepfish-prefix-dev-v1','training_seed':20261031,'source_train_sha256':sha(ANN/'train.json'),'source_5_shot_sha256':sha(ANN/'5_shot.json'),'group_hash_rule':'SHA256(UTF8("deepfish-prefix-dev-v1\\n" + prefix)); ascending digest then prefix, first <=8 eligible groups','frame_rule':'<=8 equally spaced targets across [min_frame,max_frame]; nearest available distinct frame without replacement, ties by frame then image_id; no label-content filtering','excluded_support_prefixes':sorted(excluded),'groups':group_rows,'images':[record(i) for i in selected],'extra_dev_label_budget':count(selected),'support':[record(i) for i in supports[5]['images']],'support_label_budget':count(supports[5]['images']),'dev_support_prefix_overlap':sorted({prefix(i) for i in selected}&excluded),'dev_support_habitat_overlap':sorted({habitat(i) for i in selected}&set(stats['shots']['5']['support_habitat_codes'])),'official_test':'sealed; no test JSON, pixels, evaluator or scores read in this script','total_raw_jpeg_bytes_support_and_dev':sum(record(i)['image_bytes'] for i in selected+supports[5]['images'])}
assert len(selected)<=64 and len(set(i['id'] for i in selected))==len(selected)
assert not dev['dev_support_prefix_overlap']
assert not ({i['id'] for i in selected}&{i['id'] for i in supports[5]['images']})
assert not ({record(i)['image_sha256'] for i in selected}&{record(i)['image_sha256'] for i in supports[5]['images']})
(ROOT/'reports/DEV_SELECTION.json').write_text(json.dumps(dev,indent=2))
with (ROOT/'reports/DEV_SELECTION.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=[k for k in dev['images'][0] if k!='annotation_ids']);w.writeheader();w.writerows({k:v for k,v in row.items() if k!='annotation_ids'} for row in dev['images'])
geometry=[]
for i in selected+supports[5]['images']:
 assert (i['width'],i['height'])==(1920,1080)
 for a in boxes[i['id']]:
  x,y,w,h=a['bbox'];excess=max(0,-x,-y,x+w-i['width'],y+h-i['height'])
  assert all(math.isfinite(v) for v in [x,y,w,h,a['area']])
  assert w>0 and h>0 and abs(a['area']-w*h)<1e-5 and a['category_id']==1
  if excess:geometry.append({'image_id':i['id'],'annotation_id':a['id'],'bbox':a['bbox'],'max_boundary_excess_px':excess})
contract={'coordinate_contract':'published COCO absolute xywh pixels; width1920 height1080; conversion xyxy=(x,y,x+w,y+h)','checked_images':len(selected)+len(supports[5]['images']),'checked_boxes':sum(len(boxes[i['id']]) for i in selected+supports[5]['images']),'boundary_excesses':geometry,'published_boxes_modified':False}
(ROOT/'reports/SELECTED_GEOMETRY_CHECK.json').write_text(json.dumps(contract,indent=2))
print(json.dumps({'group_summary':stats,'dev_budget':dev['extra_dev_label_budget'],'dev_groups':group_rows,'dev_support_habitat_overlap':dev['dev_support_habitat_overlap'],'jpeg_bytes':dev['total_raw_jpeg_bytes_support_and_dev'],'geometry':contract},indent=2))
