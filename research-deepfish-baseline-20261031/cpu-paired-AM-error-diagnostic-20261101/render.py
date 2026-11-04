"""Four predeclared post-hoc examples: raw / original GT / saved A / saved M."""
from portable_dev.paths import project_path as _portable_project_path
import csv,hashlib,json,pathlib
from PIL import Image,ImageDraw,ImageFont
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
run=json.loads((R/'cpu-paired-AM-20261101/evaluation/RUN_PROTOCOL.json').read_text());ann=json.loads(pathlib.Path(run['dev_annotation']['path']).read_text())
selected=json.loads((P/'reports/SELECTED_IMAGES.json').read_text());assert len(selected)==4
rows=list(csv.DictReader((P/'reports/ALL_PREDICTION_DIAGNOSTICS.csv').open()));gtrows=list(csv.DictReader((P/'reports/GT_DIAGNOSTICS.csv').open()))
fontpath='/System/Library/Fonts/Supplemental/Arial.ttf';font=ImageFont.truetype(fontpath,17);small=ImageFont.truetype(fontpath,14);title=ImageFont.truetype(fontpath,21)
manifest=[]
for selection in selected:
 iid=selection['image_id'];src=R/'repeat-seed20261101-budget25-20261101/payload/data/images'/selection['file_name'];source_hash=hashlib.sha256(src.read_bytes()).hexdigest();im=Image.open(src).convert('RGB');assert im.size==(1920,1080)
 ga=[g for g in ann['annotations'] if g['image_id']==iid];gd=[g for g in gtrows if int(g['image_id'])==iid]
 canvas=Image.new('RGB',(1920,1320),'#111c25');draw=ImageDraw.Draw(canvas);rec={'image_id':iid,'source_sha256':source_hash,'file_name':selection['file_name'],'selection':'post-hoc '+selection['selection'],'display_rule':'top10 score-ranked predictions plus best IoU candidate for every GT; no score threshold','display':{}}
 draw.text((12,8),f"Post-hoc {selection['selection']} | image {iid} | M-A per-image AP100 {selection['delta_AP100']*100:+.2f} pp | {selection['file_name']}",font=title,fill='white')
 for n,label in enumerate(['Raw image','Original GT','A saved predictions','M saved predictions']):
  xoff=(n%2)*960;yoff=42+(n//2)*590;canvas.paste(im.resize((960,540),Image.Resampling.LANCZOS),(xoff,yoff+30));draw.text((xoff+8,yoff+4),label,font=title,fill='white')
  if n>=2:
   arm='A' if n==2 else 'M';color='#ffb347' if arm=='A' else '#00d8f0';dr=sorted([r for r in rows if r['arm']==arm and int(r['image_id'])==iid],key=lambda r:int(r['rank']));keep={int(r['dt_id']) for r in dr[:10]};keep.update(int(g[arm+'_best_iou_dt_id']) for g in gd if float(g[arm+'_best_iou'])>0);shown=[r for r in dr if int(r['dt_id']) in keep];rec['display'][arm]=shown
   for d in reversed(shown):
    x,y,w,h=json.loads(d['bbox_xywh']);x=xoff+x/2;y=yoff+30+y/2;draw.rectangle((x,y,x+w/2,y+h/2),outline=color,width=2)
    tx=max(xoff,min(xoff+900,x));ty=max(yoff+30,min(yoff+550,y-17));text=f"r{d['rank']}";draw.rectangle((tx,ty,tx+small.getlength(text)+3,ty+17),fill='#111c25');draw.text((tx+1,ty),text,font=small,fill=color)
  if n>0:
   for g in ga:
    x,y,w,h=g['bbox'];x=xoff+x/2;y=yoff+30+y/2;draw.rectangle((x,y,x+w/2,y+h/2),outline='#41ff76',width=3);tx=max(xoff,min(xoff+865,x));ty=max(yoff+30,min(yoff+550,y+h/2));text=f"GT{g['id']}";draw.rectangle((tx,ty,tx+small.getlength(text)+3,ty+17),fill='#111c25');draw.text((tx+1,ty),text,font=small,fill='#41ff76')
 draw.text((12,1228),'Green: unchanged GT | Orange: A | Cyan: M | r = within-image score rank; displayed subset is for readability only',font=font,fill='white')
 for j,g in enumerate(gd):
  text=f"GT{g['gt_id']}: best IoU A/M={float(g['A_best_iou']):.3f}/{float(g['M_best_iou']):.3f}; first IoU>=.75 rank A/M={g['A_first_rank75'] or 'none'}/{g['M_first_rank75'] or 'none'}"
  draw.text((12,1253+j*22),text,font=font,fill='white')
 target=P/'overlays'/f'image_{iid}_{selection["selection"]}.png';canvas.save(target,optimize=True);rec.update(path=str(target.relative_to(P)),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size);manifest.append(rec)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==source_hash
(P/'reports/OVERLAY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
bins=list(csv.DictReader((P/'reports/ALL_IOU_BINS.csv').open()));curves=list(csv.DictReader((P/'reports/FULL_INTERPOLATED_PR_CURVES.csv').open()))
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for arm,color in [('A','#bd6b00'),('M','#008ca4')]:
 for cap in [100,300]:
  rr=[r for r in bins if r['arm']==arm and r['group']=='ALL' and r['area']=='all' and int(r['maxDets'])==cap]
  axs[0,0].plot([float(r['IoU']) for r in rr],[100*float(r['AP']) for r in rr],color=color,marker='o',linestyle='-' if cap==100 else '--',label=f'{arm} cap{cap}')
  axs[0,1].plot([float(r['IoU']) for r in rr],[100*float(r['recall']) for r in rr],color=color,marker='o',linestyle='-' if cap==100 else '--',label=f'{arm} cap{cap}')
 for col,t in enumerate([.5,.75]):
  rr=[r for r in curves if r['arm']==arm and r['area']=='all' and int(r['maxDets'])==100 and abs(float(r['IoU'])-t)<1e-8]
  axs[1,col].step([float(r['recall_grid']) for r in rr],[float(r['precision']) for r in rr],where='post',color=color,label=arm)
for ax in axs.flat:ax.grid(alpha=.2);ax.legend();ax.set_ylim(bottom=0)
axs[0,0].set(title='AP at each IoU (post-hoc common caps)',xlabel='IoU',ylabel='AP points')
axs[0,1].set(title='Recall at each IoU',xlabel='IoU',ylabel='Recall %')
for ax,t in zip(axs[1],[.5,.75]):ax.set(title=f'PR: IoU {t}, cap100',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
fig.suptitle('CPU-trained A/M | same64 dev,187 GT | saved predictions only')
fig.savefig(P/'reports/PR_IOU_DIAGNOSTICS.png',dpi=160);plt.close(fig)
print(json.dumps({'overlays':len(manifest),'overlay_bytes':sum(r['bytes'] for r in manifest),'model_calls':0}))
