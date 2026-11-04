from counting_assets import *
import concurrent.futures, struct
g=json.loads((ROOT/'sources/countgdpp-github-commit.json').read_text())['sha']
t=json.loads((ROOT/'sources/countgdpp-github-tree.json').read_text())
names=[e['path'] for e in t['tree'] if e['type']=='blob' and (e['path'].endswith(('.py','.json','.txt')) or e['path'] in ['README.md','LICENSE'])]
jobs=[(f'https://raw.githubusercontent.com/niki-amini-naieni/CountGDPlusPlus/{g}/{x}','sources/countgdpp-github/'+x) for x in names]
jobs += [('https://huggingface.co/api/models/google-bert/bert-base-uncased','sources/bert-hf-revision.json')]
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    fs={pool.submit(get,u,r):r for u,r in jobs}
    fs[pool.submit(get,'https://huggingface.co/nikigoli/CountGDPlusPlus/resolve/06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1/model.safetensors','models/model.safetensors',max_bytes=937560192,expected_sha='94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5',timeout=60)]='model.safetensors'
    b=(ROOT/'sources/pairtally-zip-tail.bin').read_bytes();i=b.rfind(b'PK\x05\x06');e=struct.unpack_from('<4s4H2IH',b,i);start=e[6];end=start+e[5]+22-1
    u='https://drive.usercontent.google.com/download?id=1TnenXS4yFicjo81NnmClfzgc8ltmmeBv&export=download&confirm=t'
    fs[pool.submit(get,u,'sources/pairtally-zip-central.bin',max_bytes=e[5]+22,headers={'Range':f'bytes={start}-{end}'},allowed_status=(206,))]='zip-central'
    for f in concurrent.futures.as_completed(fs):
        try:
            p=f.result()
            if fs[f] in ['model.safetensors','zip-central']:print('COMPLETE',fs[f],p.stat().st_size,flush=True)
        except Exception as ex:
            errors.append({'path':fs[f],'error':str(ex)})
            print('ERROR',fs[f],str(ex),flush=True)
print(json.dumps({'jobs':len(fs),'errors':errors}))
