from counting_assets import *
import concurrent.futures, struct, zlib, io, collections
from PIL import Image
u='https://drive.usercontent.google.com/download?id=1TnenXS4yFicjo81NnmClfzgc8ltmmeBv&export=download&confirm=t'
selection=json.loads((ROOT/'reports/selection_frozen.json').read_text())['images']
ann=json.loads((ROOT/'sources/pairtally/dataset/pairtally_dataset/annotations/pairtally_annotations_simple.json').read_text())
bert=json.loads((ROOT/'sources/bert-hf-revision.json').read_text())['sha']
jobs=[(f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{bert}/{x}','models/bert-base-uncased/'+x) for x in ['LICENSE','README.md','config.json','tokenizer.json','tokenizer_config.json','vocab.txt']]
jobs.append((f'https://huggingface.co/api/models/google-bert/bert-base-uncased/tree/{bert}?recursive=false&limit=1000','sources/bert-hf-tree.json'))
results=[];errors=[]
def image_job(x):
    z=x['zip_entry']; start=z['header_offset'];end=z['range_end']
    p=get(u,'data/zip_ranges/'+x['filename']+'.zip-range',max_bytes=end-start+1,headers={'Range':f'bytes={start}-{end}'},allowed_status=(206,))
    b=p.read_bytes();h=struct.unpack_from('<4s5H3I2H',b)
    assert h[0]==b'PK\x03\x04' and h[3]==z['compression_method'] and not(h[2]&1)
    n,e=h[-2:];name=b[30:30+n].decode('utf-8' if h[2]&2048 else 'cp437')
    assert name==z['filename']
    c=b[30+n+e:30+n+e+z['compressed_bytes']]
    raw=zlib.decompress(c,-15) if z['compression_method']==8 else c
    assert len(raw)==z['uncompressed_bytes'] and f'{zlib.crc32(raw)&0xffffffff:08x}'==z['crc32']
    dest=ROOT/'data/images'/x['filename'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    with Image.open(io.BytesIO(raw)) as im:
        fmt=im.format;mode=im.mode;size=im.size;orientation=im.getexif().get(274);im.verify()
    with Image.open(io.BytesIO(raw)) as im:pix=im.convert('RGB').tobytes()
    w,h=size;point_oob=[];box_oob=[];hits=[]
    for cls,field,boxes in [('A','points',x['A_boxes_xyxy']),('B','negative_points',x['B_boxes_xyxy'])]:
        points=ann[x['filename']][field];other=ann[x['filename']]['negative_points' if cls=='A' else 'points']
        point_oob += [{'class':cls,'index':i,'xy':p} for i,p in enumerate(points) if not(0<=p[0]<=w and 0<=p[1]<=h)]
        for i,b in enumerate(boxes):
            x1,y1,x2,y2=b
            if not(0<=x1<x2<=w and 0<=y1<y2<=h):box_oob.append({'class':cls,'index':i,'xyxy':b})
            hits.append({'class':cls,'box_index':i,'same_class_point_indices':[j for j,p in enumerate(points) if x1<=p[0]<=x2 and y1<=p[1]<=y2],'opposite_class_point_indices':[j for j,p in enumerate(other) if x1<=p[0]<=x2 and y1<=p[1]<=y2]})
    return {'filename':x['filename'],'pair_id':x['pair_id'],'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'zip_crc32':z['crc32'],'crc_verified':True,'width':w,'height':h,'format':fmt,'mode':mode,'exif_orientation':orientation,'decoded_rgb_sha256':hashlib.sha256(b'RGB8\0'+struct.pack('>II',w,h)+pix).hexdigest(),'point_oob':point_oob,'box_oob':box_oob,'box_point_hits':hits}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    fs={pool.submit(get,u,r):r for u,r in jobs}
    for x in selection:fs[pool.submit(image_job,x)]=x['filename']
    for f in concurrent.futures.as_completed(fs):
        try:
            r=f.result()
            if isinstance(r,dict):results.append(r)
        except Exception as ex:errors.append({'path':fs[f],'error':str(ex)})
results.sort(key=lambda x:(x['pair_id'],x['filename']))
dump('reports/image_integrity.json',{'images':results,'errors':errors,'all_model_forward_calls':0,'gpu_operations':0})
hits=[(x['filename'],h) for x in results for h in x['box_point_hits'] if len(h['same_class_point_indices'])!=1 or len(h['opposite_class_point_indices'])!=0]
dump('reports/box_point_anomalies.json',{'note':'A point-in-box test is a diagnostic only: point annotations need not be exact object centers. These anomalies require geometry review; no automatic correction or exclusion was performed.','anomalies':hits})
print(json.dumps({'images':len(results),'errors':errors,'point_oob_images':sum(bool(x['point_oob']) for x in results),'box_oob_images':sum(bool(x['box_oob']) for x in results),'box_point_anomalies':len(hits)}))
