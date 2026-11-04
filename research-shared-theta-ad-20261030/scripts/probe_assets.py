"""Read official archive metadata and fetch one SHA256-pinned safetensors asset."""
import hashlib,html.parser,io,json,pathlib,struct,time,urllib.parse,urllib.request,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
LOG=[]
def fetch(url,path,cap,headers=None,expected_sha=None,expected_size=None):
    start=time.monotonic()
    req=urllib.request.Request(url,headers=headers or {})
    with urllib.request.urlopen(req,timeout=45) as r:
        h=dict(r.headers); declared=int(h.get('Content-Length',h.get('content-length','0')))
        if declared>cap: raise RuntimeError(f'response exceeds request cap: {declared}>{cap}')
        dest=ROOT/path;tmp=dest.with_suffix(dest.suffix+'.partial');total=0;sha=hashlib.sha256()
        with tmp.open('wb') as f:
            while True:
                b=r.read(min(1048576,cap-total+1))
                if not b:break
                total+=len(b)
                if total>cap:raise RuntimeError('stream cap exceeded')
                f.write(b);sha.update(b)
        digest=sha.hexdigest()
        if expected_size is not None and total!=expected_size:raise RuntimeError('size mismatch')
        if expected_sha is not None and digest!=expected_sha:raise RuntimeError('SHA256 mismatch')
        tmp.rename(dest)
        record={'url':url,'path':path,'status':r.status,'bytes':total,'sha256':digest,'headers':h,'elapsed_seconds':time.monotonic()-start}
        LOG.append(record);(ROOT/'logs/asset-probe-downloads.json').write_text(json.dumps(LOG,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k not in ('headers','url')}),flush=True)
        return record
class Form(html.parser.HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.values={};self.active=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a['action'];self.active=True
        if self.active and tag=='input' and a.get('type')=='hidden':self.values[a['name']]=a['value']
    def handle_endtag(self,tag):
        if tag=='form':self.active=False
form=Form();form.feed((ROOT/'sources/aebad-drive-download-form.html').read_text())
assert form.action=='https://drive.usercontent.google.com/download'
assert form.values['id']=='14wkZAFFeudlg0NMFLsiGwS0E593b-lNo'
url=form.action+'?'+urllib.parse.urlencode(form.values)
(ROOT/'sources/aebad-resolved-download.json').write_text(json.dumps({'url':url,'source':'official Google Drive large-file confirmation form','permission':'public linked file; no credentials'},indent=2))
size=int(json.loads((ROOT/'sources/aebad-drive-item.json').read_text())[25][2]);assert size<=2*1024**3
try:
    record=fetch(url,'sources/aebad-zip-tail.bin',65557,{'Range':f'bytes={size-65557}-{size-1}'})
    assert record['status']==206,'Range not honored'
    assert record['headers'].get('Content-Range',record['headers'].get('content-range'))==f'bytes {size-65557}-{size-1}/{size}'
    tail=(ROOT/'sources/aebad-zip-tail.bin').read_bytes();pos=tail.rfind(b'PK\x05\x06');assert pos>=0
    eocd=struct.unpack('<4s4H2LH',tail[pos:pos+22]);_,disk,cd_disk,n_disk,n_total,cd_size,cd_offset,comment_len=eocd
    assert disk==cd_disk==0 and n_disk==n_total and n_total<65535 and cd_offset+cd_size<=size
    assert cd_size<8*1024*1024
    record=fetch(url,'sources/aebad-zip-central-directory.bin',cd_size,{'Range':f'bytes={cd_offset}-{cd_offset+cd_size-1}'},expected_size=cd_size)
    assert record['status']==206
    central=(ROOT/'sources/aebad-zip-central-directory.bin').read_bytes()
    fake_eocd=struct.pack('<4s4H2LH',b'PK\x05\x06',0,0,n_total,n_total,cd_size,0,0)
    z=zipfile.ZipFile(io.BytesIO(central+fake_eocd))
    entries=[{'path':i.filename,'compressed_bytes':i.compress_size,'uncompressed_bytes':i.file_size,'crc32':f'{i.CRC:08x}','local_header_offset':i.header_offset,'compress_type':i.compress_type,'flag_bits':i.flag_bits,'external_attr':i.external_attr,'is_dir':i.is_dir()} for i in z.infolist()]
    (ROOT/'sources/aebad-zip-manifest.json').write_text(json.dumps({'archive_bytes':size,'central_directory_offset':cd_offset,'central_directory_bytes':cd_size,'entry_count':n_total,'entries':entries},indent=2))
    print('MANIFEST',n_total,'entries',sum(x['uncompressed_bytes'] for x in entries),'uncompressed bytes',flush=True)
except Exception as e:
    (ROOT/'reports/archive-range-probe.json').write_text(json.dumps({'status':'not_verified','error':str(e),'archive_bytes_from_drive_metadata':size},indent=2));print('RANGE_PROBE_ERROR',repr(e),flush=True)
rev=json.loads((ROOT/'sources/hf-model-info.json').read_text())['sha']
fetch(f'https://huggingface.co/facebook/dinov2-small/resolve/{rev}/model.safetensors','models/dinov2-small/model.safetensors',90*1024*1024,expected_size=88249960,expected_sha='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1')
