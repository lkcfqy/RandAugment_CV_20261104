"""Pure protocol and budget contracts. Importing this module has no external effects."""
import collections,hashlib,json,math
SUPPORT_IDS=(930,1892,1900,2244,2363)
U_CHOICES=(200,)
RELEASE_RESERVE_SECONDS=180
MAX_CU=2.5
MAX_GPU_SECONDS=7200
FULL_CHECKPOINT_BYTES=692175962
PREPARATION_ALLOWANCE_SECONDS=554.525631070137
class ContractError(RuntimeError):pass
class NotAuthorized(ContractError):pass
class ResourceUndetermined(ContractError):pass
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest_bytes(value):return hashlib.sha256(value).hexdigest()
def positive(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0
def validate_context(ctx,now):
 if ctx.get('authorization',{}).get('model_run') is not True or ctx.get('authorization',{}).get('upload') is not True:
  raise NotAuthorized('This new experiment has no explicit upload/model-run authorization.')
 for key in ('quote_verified','persistent_storage_verified','external_release_guard_verified'):
  if ctx.get(key) is not True:raise ResourceUndetermined('Missing verified '+key)
 rate=ctx.get('cu_per_hour_upper_bound');start=ctx.get('allocation_started_epoch')
 if not positive(rate) or not positive(start) or start>now:raise ResourceUndetermined('Valid live CU upper-bound and allocation start required.')
 path=ctx.get('artifact_dir','')
 if not isinstance(path,str) or not path.startswith('/'):raise ResourceUndetermined('An absolute artifact staging destination is required.')
 if ctx.get('persistence_mode')=='local_cli_sha256':
  proof=ctx.get('persistence_probe',{})
  if proof.get('remote_sha256')!=proof.get('local_sha256') or len(proof.get('local_sha256',''))!=64 or not positive(proof.get('bytes')) or not positive(proof.get('download_bytes_per_second')):raise ResourceUndetermined('Verified current-allocation local download probe required.')
 elif path.startswith(('/tmp/','/content/outputs/')):raise ResourceUndetermined('A verified persistence transport is required for ephemeral staging.')
 prior_cu=ctx.get('prior_cu',0);prior_gpu=ctx.get('prior_gpu_seconds',0)
 if prior_cu<1.8088330815795528 or prior_gpu<5443.899788473129:raise ResourceUndetermined('Historical cumulative ledger cannot be reset.')
 if not isinstance(prior_cu,(int,float)) or not 0<=prior_cu<MAX_CU or not isinstance(prior_gpu,(int,float)) or not 0<=prior_gpu<MAX_GPU_SECONDS:raise ResourceUndetermined('Cumulative resource ledger invalid')
 hard=start+min(MAX_GPU_SECONDS-prior_gpu,(MAX_CU-prior_cu)*3600/rate);remaining=hard-now-RELEASE_RESERVE_SECONDS
 if remaining<=0:raise ResourceUndetermined('Resource budget exhausted, including finalization reserve.')
 return dict(hard_deadline_epoch=hard,work_remaining_seconds=remaining,elapsed_allocated_seconds=now-start,estimated_cu_so_far=(now-start)*rate/3600,max_cu=MAX_CU,max_allocated_seconds=MAX_GPU_SECONDS)
def verify_exposures(ids,updates):
 if not isinstance(updates,int) or updates<=0 or updates%5 or len(ids)!=updates or collections.Counter(ids)!={i:updates//5 for i in SUPPORT_IDS}:raise ContractError('Unequal or missing source exposures.')
 return dict(updates=updates,per_image=updates//5,counts=dict(collections.Counter(ids)))
def validate_pretrained_key_schema(missing,extra):
 logs={'bbox_head.cls_branches.'+str(i)+'.log_scale' for i in range(7)}
 ignored={'language_model.language_backbone.body.model.embeddings.position_ids','language_model.language_backbone.body.model.pooler.dense.weight','language_model.language_backbone.body.model.pooler.dense.bias'}
 if set(missing)!={'dn_query_generator.label_embedding.weight'}|logs or set(extra)!=ignored:raise ContractError('Pretrained key schema differs from pinned official config/checkpoint.')
 return {'log_scale_keys':sorted(logs),'ignored_checkpoint_keys':sorted(ignored)}
