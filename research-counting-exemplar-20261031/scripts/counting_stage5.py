from counting_assets import *
import concurrent.futures
pr=json.loads((ROOT/'sources/pairtally-commit.json').read_text())['sha']
jobs=[(f'https://raw.githubusercontent.com/bbvisual/PairTally_Benchmark/{pr}/{p}','sources/pairtally/'+p) for p in ['dataset/tools/convert_cvat_to_pairtally.py','dataset/tools/README.md']]
b=json.loads((ROOT/'sources/bert-hf-revision.json').read_text())['sha']
t=json.loads((ROOT/'sources/bert-hf-tree.json').read_text());x=next(x for x in t if x['path']=='model.safetensors')
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    fs={pool.submit(get,u,r):r for u,r in jobs}
    fs[pool.submit(get,f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{b}/model.safetensors','models/bert-base-uncased/model.safetensors',max_bytes=x['size'],expected_sha=x['lfs']['oid'],timeout=60)]='bert-model.safetensors'
    for f in concurrent.futures.as_completed(fs):
        try:
            p=f.result();print('COMPLETE',fs[f],p.stat().st_size,flush=True)
        except Exception as ex:errors.append({'path':fs[f],'error':str(ex)});print('ERROR',fs[f],str(ex),flush=True)
print(json.dumps({'errors':errors}))
