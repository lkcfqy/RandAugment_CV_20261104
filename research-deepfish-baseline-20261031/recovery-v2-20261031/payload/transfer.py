"""Stage receipts for verified local downloads; /content is explicitly ephemeral."""
import hashlib,json,pathlib,time
from contracts import canonical,ContractError,ResourceUndetermined
from lifecycle import atomic_bytes,atomic_json,emit_evidence
CHUNK_BYTES=8*1024**2
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def stage_index(out,transfer_root,chunk_bytes=CHUNK_BYTES):
 out=pathlib.Path(out);transfer_root=pathlib.Path(transfer_root);transfer_root.mkdir(parents=True,exist_ok=True);rows=[]
 # Preserve reports/predictions/logs before large checkpoints during cleanup.
 for p in sorted(out.rglob('*'),key=lambda p:(p.suffix in ('.pth','.tmp'),str(p))):
  if not p.is_file():continue
  if p.is_symlink():raise ContractError('Artifact symlink forbidden')
  rel=str(p.relative_to(out));full=digest(p);row={'path':rel,'bytes':p.stat().st_size,'sha256':full,'parts':[]}
  with p.open('rb') as f:
   for number in range((row['bytes']+chunk_bytes-1)//chunk_bytes):
    b=f.read(chunk_bytes);part=transfer_root/(full+'_'+str(number)+'.bin')
    if not part.exists():atomic_bytes(part,b)
    row['parts'].append({'file':part.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  rows.append(row)
 return rows
def validate_ack(ack,index_sha):
 if ack.get('index_sha256')!=index_sha or ack.get('all_files_atomically_persisted_and_hashed') is not True:raise ContractError('Local artifact acknowledgment missing or stale')
def persist_stage(out,phase,root,deadline):
 root=pathlib.Path(root);rows=stage_index(out,root/'transfer');receipt={'phase':phase,'files':rows};index_sha=hashlib.sha256(canonical(receipt)).hexdigest();emit_evidence('stage_ready',dict(receipt,index_sha256=index_sha))
 ack=root/'acks'/(phase+'.json')
 while not ack.exists():
  if time.monotonic()>=deadline:raise ResourceUndetermined('Local download receipt was not completed before work deadline')
  time.sleep(.2)
 validate_ack(json.loads(ack.read_text()),index_sha)
 return {'phase':phase,'index_sha256':index_sha,'files':len(rows),'local_persistence_verified':True}
