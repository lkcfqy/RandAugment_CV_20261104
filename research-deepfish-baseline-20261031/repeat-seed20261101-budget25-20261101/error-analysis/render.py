"""Scientific annotation overlays and saved-prediction plots, CPU only."""
import csv, hashlib, json, pathlib
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent
RUN=pathlib.Path('/Users/lkc/Documents/Codex/2026-10-31/task-8/deepfish-repeat-seed20261101-budget25-20261101')
ann=json.loads((RUN/'payload/data/annotations/dev64.json').read_text())
selected=json.loads((ROOT/'reports/SELECTED_IMAGES.json').read_text())
det=list(csv.DictReader((ROOT/'reports/ALL_PREDICTION_DIAGNOSTICS.csv').open()))
gt_diag=list(csv.DictReader((ROOT/'reports/GT_DIAGNOSTICS.csv').open()))
fontpath='/System/Library/Fonts/Supplemental/Arial.ttf'
font=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,17);large=ImageFont.truetype(fontpath,28)
output=[]
for selection in selected:
 iid=selection['image_id'];src=RUN/'payload/data/images'/selection['file_name'];raw=src.read_bytes();image=Image.open(src).convert('RGB');w,h=image.size
 assert (w,h)==(1920,1080)
 ga=[a for a in ann['annotations'] if a['image_id']==iid]
 gd=[r for r in gt_diag if int(r['image_id'])==iid]
 header=85;footer=360;canvas=Image.new('RGB',(2*w,h+header+footer),'#101922');draw=ImageDraw.Draw(canvas)
 rec={'image_id':iid,'source':str(src),'source_sha256':hashlib.sha256(raw).hexdigest(),'selection':selection['selection'],'display':{}}
 for ai,arm in enumerate('AB'):
  xoff=ai*w;canvas.paste(image,(xoff,header));color='#FFB347' if arm=='A' else '#00D8F0'
  rows=sorted([d for d in det if d['arm']==arm and int(d['image_id'])==iid],key=lambda d:int(d['rank']))
  keep={int(d['dt_id']) for d in rows[:10]}
  keep.update(int(g[arm+'_best_iou_dt_id']) for g in gd if float(g[arm+'_best_iou'])>0)
  shown=[d for d in rows if int(d['dt_id']) in keep];rec['display'][arm]=[int(d['dt_id']) for d in shown]
  draw.text((xoff+20,12),f"{arm}: factor {1.0 if arm=='A' else 0.8} | image {iid} | AP100={selection[arm+'_AP100']*100:.2f}",font=large,fill=color)
  draw.text((xoff+20,48),f"{selection['file_name']} | {selection['selection']} | B-A={selection['delta_AP100']*100:+.2f} pp",font=font,fill='white')
  for d in reversed(shown):
   x,y,bw,bh=json.loads(d['bbox_xywh']);box=(xoff+x,header+y,xoff+x+bw,header+y+bh)
   draw.rectangle(box,outline=color,width=2)
   label=f"r{d['rank']} p{d['pair_id'] or '-'}"
   tx=max(xoff,min(xoff+w-160,xoff+x));ty=max(header,min(header+h-21,header+y-22))
   draw.rectangle((tx,ty,tx+small.getlength(label)+5,ty+21),fill='#16222B');draw.text((tx+2,ty),label,font=small,fill=color)
  for g in ga:
   x,y,bw,bh=g['bbox'];draw.rectangle((xoff+x,header+y,xoff+x+bw,header+y+bh),outline='#42FF72',width=4)
   label=f"GT {g['id']}";tx=max(xoff,min(xoff+w-150,xoff+x));ty=max(header,min(header+h-22,header+y+bh))
   draw.rectangle((tx,ty,tx+small.getlength(label)+5,ty+21),fill='#16222B');draw.text((tx+2,ty),label,font=small,fill='#42FF72')
  yy=header+h+12
  draw.text((xoff+15,yy),'Green: original GT | Predictions: fixed top10 + best IoU candidate per GT (display only)',font=font,fill='white');yy+=30
  for n,d in enumerate(shown):
   col=n//12;row=n%12
   label=f"r{int(d['rank']):3d}  score={float(d['score']):.4f}  maxIoU={float(d['max_gt_iou']):.3f}  pair={d['pair_id'] or '-'}  GT={d['best_gt_id']}"
   draw.text((xoff+15+col*950,yy+row*23),label,font=small,fill=color)
 name=f"image_{iid}_{selection['selection']}.png";path=ROOT/'overlays'/name;canvas.save(path,optimize=True)
 rec.update(overlay=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest());output.append(rec)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==rec['source_sha256']
(ROOT/'reports/OVERLAY_MANIFEST.json').write_text(json.dumps(output,indent=2)+'\n')
bins=list(csv.DictReader((ROOT/'reports/ALL_IOU_BINS.csv').open()))
curves=list(csv.DictReader((ROOT/'reports/FULL_INTERPOLATED_PR_CURVES.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
for arm,color in [('A','#bd6b00'),('B','#008ca4')]:
 for ci,cap in enumerate([100,300]):
  rs=[r for r in bins if r['arm']==arm and r['group']=='ALL' and r['area']=='all' and int(r['maxDets'])==cap]
  axs[0,0].plot([float(r['IoU']) for r in rs],[100*float(r['AP']) for r in rs],color=color,marker='o',linestyle='-' if cap==100 else '--',label=f'{arm} cap{cap}')
  axs[0,1].plot([float(r['IoU']) for r in rs],[100*float(r['recall']) for r in rs],color=color,marker='o',linestyle='-' if cap==100 else '--',label=f'{arm} cap{cap}')
 for ci,t in enumerate([.5,.75]):
  rs=[r for r in curves if r['arm']==arm and r['area']=='all' and int(r['maxDets'])==100 and abs(float(r['IoU'])-t)<1e-8]
  axs[1,ci].step([float(r['recall_grid']) for r in rs],[float(r['precision']) for r in rs],where='post',color=color,label=arm)
for ax in axs.flat:ax.grid(alpha=.2);ax.legend();ax.set_ylim(bottom=0)
axs[0,0].set(title='AP at each IoU threshold',xlabel='IoU',ylabel='AP (points)')
axs[0,1].set(title='Recall at each IoU threshold',xlabel='IoU',ylabel='Recall (%)')
for ax,t in zip(axs[1],[.5,.75]):ax.set(title=f'Interpolated PR, IoU={t}, maxDets=100',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
fig.suptitle('Post-hoc diagnostics only | 64 fixed dev images, 187 GT | A=1.0, B=0.8')
fig.savefig(ROOT/'reports/PR_IOU_DIAGNOSTICS.png',dpi=170);fig.savefig(ROOT/'reports/PR_IOU_DIAGNOSTICS.pdf');plt.close(fig)
print(json.dumps({'overlays':len(output),'overlay_bytes':sum(x['bytes'] for x in output)},indent=2))
regions=[
 ('1479 top4 small unannotated object','7398_F5_f000252.jpg',(1030,290,1220,440)),
 ('1479 top2 B bottom object (leaf-like)','7398_F5_f000252.jpg',(530,700,850,870)),
 ('1481 GT5089','7398_F5_f000262.jpg',(190,580,360,730)),
 ('1149 adjoining GT3828/3830','7117_Caranx_sexfasciatus_juvenile_f000375.jpg',(1500,220,1700,430))]
detail=Image.new('RGB',(1000,900),'#15212c');dd=ImageDraw.Draw(detail)
for n,(label,fn,box) in enumerate(regions):
 im=Image.open(RUN/'payload/data/images'/fn).crop(box);im.thumbnail((480,380))
 if im.width*2<480 and im.height*2<380:im=im.resize((im.width*2,im.height*2))
 col=n%2;row=n//2;detail.paste(im,(col*500+10,row*450+40));dd.text((col*500+10,row*450+12),label,fill='white')
detail.save(ROOT/'overlays/selected_detail_crops.png')
(ROOT/'reports/DETAIL_CROPS.json').write_text(json.dumps({'note':'Post-view zooms only within the already frozen six examples; no new example or metric selection. Labels are visual impressions, not relabeling.','regions':[{'description':a,'file_name':b,'crop_xyxy':c} for a,b,c in regions]},indent=2)+'\n')
