"""Fetch only three named public metadata/PDF resources; no credentials, code execution, or dependencies."""
import hashlib,json,pathlib,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1]
targets=[
 ('original_paper.pdf','https://researchonline.jcu.edu.au/64493/1/64493.pdf',8*1024**2),
 ('original_archive_index.html','https://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/',1024**2),
 ('mmcv_cu117_torch200_index.html','https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html',1024**2)
]
rows=[]
for name,url,limit in targets:
 row={'url':url,'target':name,'maximum_bytes':limit,'attempts':1,'performed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DeepFish-asset-audit/1.0'}),timeout=40) as response:
   row.update(status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'))
   data=response.read(limit+1)
   if len(data)>limit:raise ValueError('Metadata/PDF byte ceiling exceeded')
   (ROOT/'sources'/name).write_bytes(data)
   row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),outcome='downloaded')
 except Exception as e:row.update(outcome='failed',reason=type(e).__name__+': '+str(e))
 rows.append(row)
 (ROOT/'reports/PUBLIC_FETCH_LOG.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps(row))
