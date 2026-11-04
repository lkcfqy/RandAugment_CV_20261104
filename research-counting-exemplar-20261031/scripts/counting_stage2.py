from counting_assets import *
import concurrent.futures
p=json.loads((ROOT/'sources/pairtally-commit.json').read_text())['sha']
g=json.loads((ROOT/'sources/countgdpp-github-commit.json').read_text())['sha']
t=json.loads((ROOT/'sources/pairtally-tree.json').read_text())
names=[e['path'] for e in t['tree'] if e['type']=='blob' and (e['path'] in ['README.md','LICENSE','DOWNLOAD_SETUP.md','dataset/README.md','dataset/verify_dataset.py'] or e['path'].startswith('dataset/pairtally_dataset/annotations/') or e['path'].startswith('models/countgd_plusplus/'))]
jobs=[(f'https://raw.githubusercontent.com/bbvisual/PairTally_Benchmark/{p}/{x}','sources/pairtally/'+x) for x in names]
jobs += [(f'https://api.github.com/repos/niki-amini-naieni/CountGDPlusPlus/git/trees/{g}?recursive=1','sources/countgdpp-github-tree.json'),('https://huggingface.co/api/models/nikigoli/CountGDPlusPlus/tree/06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1?recursive=true&limit=1000','sources/countgdpp-hf-tree-all.json'),('https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view','sources/pairtally-drive-page.html')]
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    fs={pool.submit(get,u,r):r for u,r in jobs}
    for f in concurrent.futures.as_completed(fs):
        try:f.result()
        except Exception as ex:errors.append({'path':fs[f],'error':str(ex)})
print(json.dumps({'jobs':len(jobs),'errors':errors}))
u='https://drive.usercontent.google.com/download?id=1TnenXS4yFicjo81NnmClfzgc8ltmmeBv&export=download&confirm=t'
try:
    p=get(u,'sources/pairtally-zip-tail.bin',max_bytes=200_000,headers={'Range':'bytes=-65557'},allowed_status=(200,206))
    print('range probe bytes',p.stat().st_size,'prefix',p.read_bytes()[:50])
except Exception as ex:print('range probe:',str(ex))
