"""Decode six declared images for annotation QA, with published boxes only."""
import collections,hashlib,io,json,pathlib,re,tarfile
from PIL import Image,ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[1]
ANN=ROOT/'assets/annotation_docs/FISH/annotations'
datasets={s:json.loads((ANN/(s+'.json')).read_text()) for s in ['1_shot','5_shot','10_shot','test']}
audit=json.loads((ROOT/'reports/ANNOTATION_AUDIT.json').read_text())
chosen=[]
for s in ['1_shot','5_shot','10_shot','test']:
 chosen.append({'split':s,'reason':'lowest numeric image ID in supplied split','image':min(datasets[s]['images'],key=lambda x:x['id'])})
edge=audit['splits']['test']['geometry_issues'][0]['image_id']
chosen.append({'split':'test','reason':'first annotation-order image with a boundary-overrun flag','image':next(x for x in datasets['test']['images'] if x['id']==edge)})
support=datasets['1_shot']['images'][0];pref=re.sub(r'_f\d+\.[^.]+$','',support['file_name']);frame=int(re.search(r'_f(\d+)',support['file_name']).group(1))
candidates=[x for x in datasets['test']['images'] if re.sub(r'_f\d+\.[^.]+$','',x['file_name'])==pref]
nearest=min(candidates,key=lambda x:(abs(int(re.search(r'_f(\d+)',x['file_name']).group(1))-frame),x['id']))
chosen.append({'split':'test','reason':'nearest filename frame number to 1-shot support in same prefix group, metadata heuristic only','image':nearest})
for r in chosen:
 r['member']='FISH/'+('test' if r['split']=='test' else 'train')+'/'+r['image']['file_name']
 r['annotations']=[a for a in datasets[r['split']]['annotations'] if a['image_id']==r['image']['id']]
(ROOT/'reports/PREVIEW_PLAN.json').write_text(json.dumps(chosen,indent=2))
manifest={r['name']:r for r in json.loads((ROOT/'reports/FISH_archive_members.json').read_text())}
selected={r['member'] for r in chosen};raw={}
with tarfile.open(ROOT/'assets/FISH.tar.gz','r|gz') as tf:
 for m in tf:
  if m.name not in selected:continue
  if not m.isfile():raise ValueError('Preview member not regular')
  b=tf.extractfile(m).read()
  assert hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256']
  raw[m.name]=b
  p=ROOT/'preview/originals'/pathlib.PurePosixPath(m.name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  if len(raw)==len(selected):break
sheet=Image.new('RGB',(1920,3*600),(20,25,30));draw=ImageDraw.Draw(sheet)
checks=[]
for i,r in enumerate(chosen):
 im=Image.open(io.BytesIO(raw[r['member']])).convert('RGB');im.load()
 info={'member':r['member'],'decoded_size':list(im.size),'declared_size':[r['image']['width'],r['image']['height']],'matches':im.size==(r['image']['width'],r['image']['height']),'annotation_count':len(r['annotations'])};checks.append(info)
 dd=ImageDraw.Draw(im)
 for a in r['annotations']:
  x,y,w,h=a['bbox'];dd.rectangle((x,y,x+w,y+h),outline=(80,255,70),width=4)
 im=im.resize((960,540));ox=(i%2)*960;oy=(i//2)*600;sheet.paste(im,(ox,oy+60))
 draw.text((ox+12,oy+7),f'{r["split"]} | image ID {r["image"]["id"]} | {len(r["annotations"])} published boxes',fill='white')
 draw.text((ox+12,oy+28),r['image']['file_name'],fill='white')
sheet.save(ROOT/'preview/annotation_contact_sheet.jpg',quality=94)
(ROOT/'reports/PREVIEW_DECODE_CHECK.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
