"""Static ZIP/Mach-O metadata only; never import/install/load wheel code."""
import urllib.request,urllib.parse,pathlib,json,hashlib,io,zipfile,struct,datetime
W=pathlib.Path(__file__).resolve().parent
url='https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp311-cp311-macosx_10_9_universal2.whl'
req=urllib.request.Request(url,method='HEAD')
with urllib.request.urlopen(req,timeout=25) as r:head=dict(r.headers);size=int(r.headers.get('Content-Length') or 0)
rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':url,'HEAD':head,'announced_bytes':size,'installed':False,'binary_loaded_or_executed':False,'model_instantiated':False}
limit=16*1024**2
if size<=0 or size>limit:
 rec['status']='NOT_DOWNLOADED_SIZE_UNKNOWN_OR_EXCEEDS16MiB'
else:
 with urllib.request.urlopen(url,timeout=30) as r:data=r.read(limit+1)
 assert len(data)==size and len(data)<=limit
 p=W/'sources/mmcv-2.1.0-cp311-universal2.inspection-only.whl.bin';p.write_bytes(data)
 rec.update(status='STATIC_WHEEL_INSPECTED',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),path=str(p.relative_to(W)))
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
  rec['wheel_metadata']=z.read('mmcv-2.1.0.dist-info/WHEEL').decode()
  rec['extensions']=[]
  for n in z.namelist():
   if n.endswith(('.so','.dylib')):
    with z.open(n) as f:h=f.read(128)
    magic=h[:4].hex();arch=[]
    if magic in ['cafebabe','cafebabf']:
     count=struct.unpack('>I',h[4:8])[0];stride=32 if magic=='cafebabf' else 20
     for i in range(count):
      cpu,sub=struct.unpack('>II',h[8+i*stride:16+i*stride]);arch.append({'cputype_hex':hex(cpu),'cpusubtype_hex':hex(sub),'name':{0x1000007:'x86_64',0x100000c:'arm64'}.get(cpu,'unknown')})
    elif magic=='cffaedfe':
     cpu,sub=struct.unpack('<II',h[4:12]);arch.append({'cputype_hex':hex(cpu),'cpusubtype_hex':hex(sub),'name':{0x1000007:'x86_64',0x100000c:'arm64'}.get(cpu,'unknown')})
    rec['extensions'].append({'path':n,'uncompressed_bytes':z.getinfo(n).file_size,'magic':magic,'architectures':arch})
  meta=z.read('mmcv-2.1.0.dist-info/METADATA').decode();rec['metadata_dependencies']=[x for x in meta.splitlines() if x.startswith(('Requires-Dist:','Requires-Python:','Version:'))]
  rec['required_ext_present']=any(x['path'].startswith('mmcv/_ext') for x in rec['extensions'])
  rec['arm64_extension_present']=any(any(a['name']=='arm64' for a in x['architectures']) for x in rec['extensions'] if x['path'].startswith('mmcv/_ext'))
(W/'OFFICIAL_WHEEL_INSPECTION.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps({k:v for k,v in rec.items() if k!='HEAD'},indent=2))
