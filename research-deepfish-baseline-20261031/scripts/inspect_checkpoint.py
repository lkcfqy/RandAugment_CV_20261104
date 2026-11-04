"""CPU-only safe state-dict inspection. No model construction and no pickle fallback."""
import datetime,hashlib,json,pathlib,sys,traceback
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
path=ROOT/'assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
report={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(path.relative_to(ROOT)),'torch_version':torch.__version__,'python':sys.version,'weights_only':True,'map_location':'cpu','model_instantiated':False,'upstream_code_executed':False,'unsafe_fallback':False,'dynamic_whitelist':False}
h=hashlib.sha256();md5=hashlib.md5()
with path.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);md5.update(b)
report.update(bytes=path.stat().st_size,sha256=h.hexdigest(),md5=md5.hexdigest(),expected_bytes=691901857,expected_etag_md5='aa587e4dde3f2de265f33593bc9de6c4')
assert report['bytes']==report['expected_bytes']
assert report['md5']==report['expected_etag_md5']
try:
 torch.set_num_threads(2)
 checkpoint=torch.load(path,map_location='cpu',weights_only=True)
 report['top_level_type']=type(checkpoint).__name__
 report['top_level_keys']=list(checkpoint) if isinstance(checkpoint,dict) else None
 state=checkpoint.get('state_dict',checkpoint) if isinstance(checkpoint,dict) else checkpoint
 if not isinstance(state,dict): raise TypeError('Not a state dict')
 tensor_rows=[];other=[]
 for k,t in state.items():
  if not isinstance(t,torch.Tensor):other.append({'key':str(k),'type':type(t).__name__});continue
  if t.device.type!='cpu': raise RuntimeError('Non-CPU tensor')
  finite=bool(torch.isfinite(t).all().item()) if t.is_floating_point() or t.is_complex() else True
  tensor_rows.append({'key':k,'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'numel':t.numel(),'finite':finite})
 report.update(status='inspected',tensor_count=len(tensor_rows),total_tensor_elements=sum(x['numel'] for x in tensor_rows),all_finite=all(x['finite'] for x in tensor_rows),non_tensor_state_entries=other,tensors=tensor_rows)
except Exception as e:
 report.update(status='safe_load_blocked_no_fallback',error_type=type(e).__name__,error=str(e))
(ROOT/'reports/WEIGHT_CHECK.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['tensors','top_level_keys']},indent=2))
