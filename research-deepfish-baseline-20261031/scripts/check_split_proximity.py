import bisect,collections,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];ANN=ROOT/'assets/annotation_docs/FISH/annotations'
data={s:json.loads((ANN/(s+'.json')).read_text()) for s in ['train','test','1_shot','5_shot','10_shot']}
def parse(fn):
 m=re.fullmatch(r'(.+)_f(\d+)\.jpg',fn)
 if m is None:raise ValueError(fn)
 return m[1],int(m[2])
groups={}
for s,d in data.items():
 g=collections.defaultdict(list)
 for im in d['images']:
  pref,n=parse(im['file_name']);g[pref].append((n,im))
 groups[s]={k:sorted(v,key=lambda x:x[0]) for k,v in g.items()}
rows=[]
for im in data['test']['images']:
 pref,n=parse(im['file_name']);candidates=groups['train'].get(pref,[])
 if not candidates:continue
 near=min(candidates,key=lambda x:(abs(x[0]-n),x[0]))
 rows.append({'test_id':im['id'],'test_file':im['file_name'],'nearest_train_id':near[1]['id'],'nearest_train_file':near[1]['file_name'],'filename_frame_distance':abs(n-near[0])})
support=[]
for s in ['1_shot','5_shot','10_shot']:
 for im in data[s]['images']:
  pref,n=parse(im['file_name']);near=min(groups['test'][pref],key=lambda x:(abs(x[0]-n),x[0]))
  support.append({'split':s,'support_id':im['id'],'support_file':im['file_name'],'nearest_test_id':near[1]['id'],'nearest_test_file':near[1]['file_name'],'filename_frame_distance':abs(n-near[0])})
report={'semantics':'filename adjacency only; not a verified temporal interval or site mapping','test_to_full_train_distance_histogram':dict(sorted(collections.Counter(x['filename_frame_distance'] for x in rows).items())),'test_to_full_train_pairs':rows,'support_to_test_pairs':support}
(ROOT/'reports/FILENAME_PROXIMITY.json').write_text(json.dumps(report,indent=2));print(json.dumps({'histogram':report['test_to_full_train_distance_histogram'],'support_distances':[(x['split'],x['support_id'],x['filename_frame_distance']) for x in support]},indent=2))
