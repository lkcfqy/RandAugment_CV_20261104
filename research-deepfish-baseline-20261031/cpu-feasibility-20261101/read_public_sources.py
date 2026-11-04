"""Bounded official documentation/metadata GETs only. No distributions installed or executed."""
import pathlib,urllib.request,urllib.parse,hashlib,json,datetime,concurrent.futures
W=pathlib.Path(__file__).resolve().parent
raw='https://raw.githubusercontent.com/open-mmlab/'
urls={
 'mmcv_build.md':raw+'mmcv/v2.1.0/docs/en/get_started/build.md',
 'mmcv_setup.py.txt':raw+'mmcv/v2.1.0/setup.py',
 'mmcv_msda.py.txt':raw+'mmcv/v2.1.0/mmcv/ops/multi_scale_deform_attn.py',
 'mmcv_msda_tests.py.txt':raw+'mmcv/v2.1.0/tests/test_ops/test_ms_deformable_attn.py',
 'mmcv_photometric.py.txt':raw+'mmcv/v2.1.0/mmcv/image/photometric.py',
 'mmdet_get_started.md':raw+'mmdetection/v3.3.0/docs/en/get_started.md',
 'mmdet_grounding_dino_layers.py.txt':raw+'mmdetection/v3.3.0/mmdet/models/layers/transformer/grounding_dino_layers.py',
 'mmdet_deformable_detr_layers.py.txt':raw+'mmdetection/v3.3.0/mmdet/models/layers/transformer/deformable_detr_layers.py',
 'mmdet_grounding_dino_head.py.txt':raw+'mmdetection/v3.3.0/mmdet/models/dense_heads/grounding_dino_head.py',
 'mmdet_bert.py.txt':raw+'mmdetection/v3.3.0/mmdet/models/language_models/bert.py',
 'mmdet_colorspace.py.txt':raw+'mmdetection/v3.3.0/mmdet/datasets/transforms/colorspace.py',
 'mmdet_augment_wrappers.py.txt':raw+'mmdetection/v3.3.0/mmdet/datasets/transforms/augment_wrappers.py',
 'mmdet_init.py.txt':raw+'mmdetection/v3.3.0/mmdet/__init__.py',
 'openmmlab_cpu_torch210_index.html':'https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html',
}
packages={'torch':'2.1.0','torchvision':'0.16.0','mmcv':'2.1.0','numpy':'1.24.3','opencv-python':'4.11.0.86','pycocotools':'2.0.8','fairscale':'0.4.13','ninja':'1.11.1.1','psutil':'5.9.8','pip':'23.3.2','wheel':'0.43.0'}
for p,v in packages.items():urls[f'pypi_{p}_{v}.json']=f'https://pypi.org/pypi/{p}/{v}/json'
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'DeepFish-static-compatibility-inspection/1.0'})
  with urllib.request.urlopen(req,timeout=25) as r:
   assert urllib.parse.urlparse(r.url).hostname in ['raw.githubusercontent.com','pypi.org','download.openmmlab.com']
   data=r.read(700001);assert len(data)<=700000;ctype=r.headers.get('Content-Type')
  (W/'sources'/name).write_bytes(data)
  return {'file':name,'url':url,'status':'READ_OK','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'content_type':ctype}
 except Exception as e:return {'file':name,'url':url,'status':'READ_FAILED','error':type(e).__name__+': '+str(e),'retry_performed':False}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,urls.items()))
meta=[]
for p,v in packages.items():
 path=W/'sources'/f'pypi_{p}_{v}.json'
 if not path.exists():continue
 d=json.loads(path.read_text());fs=[]
 for f in d['urls']:
  n=f['filename']
  if f['packagetype']=='sdist' or ('macosx' in n and ('arm64' in n or 'universal2' in n) and ('cp39' in n or 'abi3' in n or 'py3-' in n)) or ('py3-none-any.whl' in n):
   fs.append({k:f[k] for k in ['filename','size','url','digests','requires_python','yanked']})
 meta.append({'package':p,'version':v,'requires_python':d['info']['requires_python'],'all_release_file_count':len(d['urls']),'selected_candidate_files':fs})
(W/'SOURCE_MANIFEST.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'action':'official documentation and package metadata GET only','installed':False,'model_instantiated':False,'GPU_allocated':False,'total_download_bytes':sum(x.get('bytes',0) for x in rows),'sources':rows},indent=2)+'\n')
(W/'PACKAGE_METADATA.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'source_count':len(rows),'total_text_metadata_bytes':sum(x.get('bytes',0) for x in rows),'failed':[x for x in rows if x['status']!='READ_OK'],'candidate_distributions':[{k:x[k] for k in ['package','version','requires_python','all_release_file_count']}|{'filenames':[f['filename'] for f in x['selected_candidate_files']]} for x in meta]},indent=2))
