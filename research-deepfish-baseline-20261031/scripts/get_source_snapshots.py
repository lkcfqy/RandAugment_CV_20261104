import json
from fetch_public import fetch
for repo,name in [('LiYu0524/Domain-RAG','Domain-RAG'),('lovelyqian/CDFSOD-benchmark','CDFSOD-benchmark')]:
    meta=fetch('https://api.github.com/repos/'+repo+'/commits/main','sources/'+name+'-commit.json',2*1024**2)
    sha=json.loads(meta.read_text())['sha']
    fetch('https://codeload.github.com/'+repo+'/tar.gz/'+sha,'sources/'+name+'-'+sha+'.tar.gz',50*1024**2,300)
fetch('https://huggingface.co/datasets/Alzayats/DeepFish/raw/main/README.md','sources/DeepFish-HF-README.md',1024**2)
fetch('https://drive.google.com/drive/folders/16SDv_V7RDjTKDk8uodL2ubyubYTMdd5q','sources/official-drive-folder.html',5*1024**2)
fetch('https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth','sources/weight-head.txt',32768,60,head=True)
