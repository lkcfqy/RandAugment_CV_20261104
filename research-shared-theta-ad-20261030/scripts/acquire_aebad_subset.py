"""Acquire predeclared AeBAD-S paths using official HTTP byte ranges; no image/model analysis."""
import collections,hashlib,json,pathlib,stat,struct,time,urllib.error,urllib.request,zlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'sources/aebad-zip-manifest.json').read_text());ENTRIES=D['entries'];SIZE=D['archive_bytes']
URL=json.loads((ROOT/'sources/aebad-resolved-download.json').read_text())['url']
selected=[];counts=collections.Counter()
for e in ENTRIES:
    p=pathlib.PurePosixPath(e['path']);parts=p.parts
    if len(parts)!=6 or parts[:2]!=('AeBAD','AeBAD_S') or p.suffix!='.png' or p.name.startswith('.'):continue
    _,_,split,label,domain,name=parts
    if (split=='train' and label=='good') or (split in ('test','ground_truth') and domain in ('same','illumination')):
        assert not p.is_absolute() and '..' not in parts and not stat.S_ISLNK(e['external_attr']>>16)
        assert not(e['flag_bits']&1) and e['compress_type'] in (0,8)
        selected.append(e);counts[f'{split}/{domain}/{"normal" if label=="good" else "anomaly"}']+=1
assert sum(v for k,v in counts.items() if k.startswith('train/'))==521
assert counts['test/same/normal']==230 and counts['test/same/anomaly']==459
assert counts['test/illumination/normal']==75 and counts['test/illumination/anomaly']==198
assert sum(v for k,v in counts.items() if k.startswith('ground_truth/'))==657
ordered=sorted(ENTRIES,key=lambda e:e['local_header_offset'])
nextoff={e['local_header_offset']:(ordered[i+1]['local_header_offset'] if i+1<len(ordered) else D['central_directory_offset']) for i,e in enumerate(ordered)}
groups=collections.defaultdict(list)
for e in selected:groups[str(pathlib.PurePosixPath(e['path']).parent)].append(e)
plan=[]
for folder,g in sorted(groups.items()):
    start=min(e['local_header_offset'] for e in g);end=max(nextoff[e['local_header_offset']] for e in g)-1
    plan.append({'folder':folder,'start':start,'end':end,'bytes':end-start+1,'files':len(g),'uncompressed_bytes':sum(e['uncompressed_bytes'] for e in g),'entries':g})
plan_bytes=sum(x['bytes'] for x in plan)
assert plan_bytes+150*1024**2<2*1024**3
assert plan_bytes+sum(e['uncompressed_bytes'] for e in selected)+400*1024**2<3*1024**3
(ROOT/'sources/aebad-selected-manifest.json').write_text(json.dumps({'selection_rule':'all train/good PNG; test and ground_truth same/illumination PNG; omit dot files and __MACOSX','counts':counts,'range_download_bytes':plan_bytes,'image_and_mask_bytes':sum(e['uncompressed_bytes'] for e in selected),'entries':selected},indent=2))
(ROOT/'sources/aebad-range-plan.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='entries'} for x in plan],indent=2))
(ROOT/'data/archive-ranges').mkdir(exist_ok=True)
ledger=[];downloaded=0;start_time=time.monotonic()
for number,group in enumerate(plan):
    assert time.monotonic()-start_time<1200,'bounded acquisition time budget'
    a,b=group['start'],group['end'];target=ROOT/'data/archive-ranges'/f'{a}-{b}.bin'
    last_error=None
    for attempt in range(2):
        count=0;sha=hashlib.sha256();entry={'folder':group['folder'],'attempt':attempt+1,'range':f'bytes={a}-{b}','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        try:
            req=urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}'})
            with urllib.request.urlopen(req,timeout=45) as response:
                assert response.status==206,'HTTP Range not honored; refusing whole response'
                assert response.headers.get('Content-Range')==f'bytes {a}-{b}/{SIZE}','Unexpected Content-Range'
                assert int(response.headers['Content-Length'])==group['bytes']
                entry['status']=response.status;entry['headers']=dict(response.headers)
                with target.with_suffix('.partial').open('wb') as out:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        count+=len(chunk);downloaded+=len(chunk)
                        assert count<=group['bytes'] and downloaded+150*1024**2<=2*1024**3
                        out.write(chunk);sha.update(chunk)
                assert count==group['bytes'],'Truncated range'
                target.with_suffix('.partial').rename(target)
            entry.update({'received_bytes':count,'sha256':sha.hexdigest(),'saved_path':str(target.relative_to(ROOT))})
            ledger.append(entry);(ROOT/'logs/aebad-acquisition.json').write_text(json.dumps(ledger,indent=2))
            last_error=None;break
        except Exception as err:
            entry.update({'received_bytes':count,'error':repr(err)});ledger.append(entry)
            (ROOT/'logs/aebad-acquisition.json').write_text(json.dumps(ledger,indent=2));last_error=err
            if isinstance(err,urllib.error.HTTPError) and err.code in (401,403,404,429):raise
            if not isinstance(err,(TimeoutError,urllib.error.URLError,ConnectionError)):raise
    if last_error:raise last_error
    # Local file headers, central sizes, per-member CRC and exact paths are all checked.
    with target.open('rb') as source:
        for e in group['entries']:
            source.seek(e['local_header_offset']-a);head=source.read(30)
            h=struct.unpack('<4s5H3L2H',head);assert h[0]==b'PK\x03\x04' and h[3]==e['compress_type']
            name=source.read(h[-2]).decode('utf-8' if h[2]&0x800 else 'cp437');assert name==e['path']
            source.read(h[-1]);compressed=source.read(e['compressed_bytes']);assert len(compressed)==e['compressed_bytes']
            if e['compress_type']==0:raw=compressed
            else:
                z=zlib.decompressobj(-15);raw=z.decompress(compressed,e['uncompressed_bytes']+1)
                assert z.eof and not z.unused_data and not z.unconsumed_tail
            assert len(raw)==e['uncompressed_bytes'] and f'{zlib.crc32(raw)&0xffffffff:08x}'==e['crc32']
            dest=ROOT/'data'/e['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    print(json.dumps({'group':number+1,'of':len(plan),'folder':group['folder'],'files':group['files'],'download_bytes':downloaded,'elapsed_seconds':round(time.monotonic()-start_time,2)}),flush=True)
report={'status':'complete','source_file_id':'14wkZAFFeudlg0NMFLsiGwS0E593b-lNo','downloaded_range_bytes':downloaded,'image_files':1483,'mask_files':657,'selected_png_files':len(selected),'elapsed_seconds':time.monotonic()-start_time,'all_member_crc_verified':True,'whole_archive_downloaded':False,'whole_archive_sha256':'not available: partial official range retrieval'}
(ROOT/'reports/data-acquisition.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
