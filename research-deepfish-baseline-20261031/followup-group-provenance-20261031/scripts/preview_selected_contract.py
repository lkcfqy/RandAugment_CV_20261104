"""Inspect 5 support + 8 fixed dev representatives with existing boxes. No model, no test reads."""
import collections,hashlib,io,json,pathlib,sys,tarfile
from PIL import Image,ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[1];ASSET=pathlib.Path(sys.argv[1])
dev=json.loads((ROOT/'reports/DEV_SELECTION.json').read_text());train=json.loads((ASSET/'assets/annotation_docs/FISH/annotations/train.json').read_text())
anns=collections.defaultdict(list)
for a in train['annotations']:anns[a['image_id']].append(a)
selected=[dict(r,role='support') for r in dev['support']]
for group in dev['groups']:
 selected.append(dict(next(i for i in dev['images'] if i['image_id']==group['selected_image_ids'][0]),role='dev'))
need={r['archive_member']:r for r in selected};raw={};checks=[]
(ROOT/'preview/originals').mkdir(parents=True,exist_ok=True)
(ROOT/'preview/overlays').mkdir(parents=True,exist_ok=True)
with tarfile.open(ASSET/'assets/FISH.tar.gz','r|gz') as archive:
 for member in archive:
  if member.name not in need:continue
  assert member.isfile() and member.size<10*1024**2
  r=need[member.name];data=archive.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==r['image_sha256'];raw[member.name]=data
  (ROOT/'preview/originals'/r['file_name']).write_bytes(data)
  if len(raw)==len(need):break
assert set(raw)==set(need)
for role in ('support','dev'):
 rows=[r for r in selected if r['role']==role];sheet=Image.new('RGB',(1920,600*((len(rows)+1)//2)),(22,25,30));sd=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows):
  im=Image.open(io.BytesIO(raw[r['archive_member']]));im.load();assert im.size==(r['width'],r['height']);original_mode=im.mode;im=im.convert('RGB');d=ImageDraw.Draw(im)
  for a in anns[r['image_id']]:
   x,y,w,h=a['bbox'];d.rectangle((x,y,x+w,y+h),outline=(80,255,80),width=4);d.text((max(0,x),max(0,y-16)),str(a['id']),fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
  im.save(ROOT/'preview/overlays'/(str(r['image_id'])+'.png'))
  ox=960*(j%2);oy=600*(j//2);sheet.paste(im.resize((960,540)),(ox,oy+60));sd.text((ox+10,oy+8),f'{role} | id {r["image_id"]} | {len(anns[r["image_id"]])} published boxes',fill='white');sd.text((ox+10,oy+30),r['file_name'],fill='white')
  checks.append({'image_id':r['image_id'],'role':role,'file_name':r['file_name'],'sha256':r['image_sha256'],'decoded_size':list(im.size),'decoded_mode':original_mode,'boxes':len(anns[r['image_id']]),'modified_annotation':False})
 sheet.save(ROOT/'preview'/(role+'_contract_sheet.jpg'),quality=94)
(ROOT/'reports/DECODED_CONTRACT_SAMPLES.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
