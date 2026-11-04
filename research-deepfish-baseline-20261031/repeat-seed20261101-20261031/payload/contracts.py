"""Pure protocol and budget contracts. Importing this module has no external effects."""
import collections,hashlib,json,math
SUPPORT_IDS=(930,1892,1900,2244,2363)
U_CHOICES=(25,50,100,200)
RELEASE_RESERVE_SECONDS=180
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
 if not isinstance(prior_cu,(int,float)) or not 0<=prior_cu<2 or not isinstance(prior_gpu,(int,float)) or not 0<=prior_gpu<7200:raise ResourceUndetermined('Cumulative resource ledger invalid')
 hard=start+min(7200-prior_gpu,(2.0-prior_cu)*3600/rate);remaining=hard-now-RELEASE_RESERVE_SECONDS
 if remaining<=0:raise ResourceUndetermined('Resource budget exhausted, including finalization reserve.')
 return dict(hard_deadline_epoch=hard,work_remaining_seconds=remaining,elapsed_allocated_seconds=now-start,estimated_cu_so_far=(now-start)*rate/3600,max_cu=2.0,max_allocated_seconds=7200)
def choose_updates(timings,remaining_seconds):
 fields={'init_seconds','p90_update_a','p90_update_b','p90_predict_seconds','checkpoint_bytes','write_bps'}
 if set(timings)!=fields or not all(positive(v) for v in timings.values()) or not positive(remaining_seconds):raise ContractError('Only finite positive timing/cost fields may select U; no loss or metric inputs.')
 t=timings;estimates={}
 for u in U_CHOICES:
  estimates[str(u)]=2*t['init_seconds']+1.5*u*(t['p90_update_a']+t['p90_update_b'])+1.5*128*t['p90_predict_seconds']+4*t['checkpoint_bytes']/t['write_bps']+120
 fit=[u for u in U_CHOICES if estimates[str(u)]<=remaining_seconds]
 return dict(status='FROZEN' if fit else 'RESOURCE_UNDETERMINED',updates=max(fit) if fit else 0,estimated_seconds=estimates[str(max(fit))] if fit else None,estimates_seconds=estimates,remaining_seconds=remaining_seconds,rule='largest fixed candidate fitting conservative timing cost; no efficacy input')
def fixed_repeat_updates(timings,remaining_seconds):
 result=choose_updates(timings,remaining_seconds);fits=result['estimates_seconds']['200']<=remaining_seconds
 result.update(status='FROZEN' if fits else 'RESOURCE_UNDETERMINED',updates=200 if fits else 0,estimated_seconds=result['estimates_seconds']['200'],rule='previously fixed U200 or no run; no smaller-U fallback',seed=20261101,post_result_exploratory_repeat=True,previous_seed_dev_metrics_seen=True,new_seed_dev_metrics_seen=False)
 return result
def repeat_training_seed(config):
 seed=config.get('randomness',{}).get('seed')
 if type(seed) is not int or seed!=20261101:raise ContractError('Only the authorized repeat training seed20261101 is allowed.')
 return seed
def require_complete_repeat_budget(prior_cu,prior_gpu,rate,elapsed):
 if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (prior_cu,prior_gpu,rate,elapsed)) or not 0<=prior_cu<2 or not 0<=prior_gpu<7200 or rate<=0 or elapsed<0:raise ResourceUndetermined('Invalid live cumulative budget.')
 remaining=min(7200-prior_gpu,(2-prior_cu)*3600/rate)-elapsed;required=2320.621211230755
 if remaining<required:raise ResourceUndetermined('Complete fixed-U repeat including persistence and release reserve does not fit.')
 return dict(required_seconds=required,remaining_seconds=remaining,live_rate=rate,prior_cu=prior_cu,prior_gpu_seconds=prior_gpu,paired_seed_count=1,seed=20261101,formal_U=200)
def verify_exposures(ids,updates):
 if not isinstance(updates,int) or updates<=0 or updates%5 or len(ids)!=updates or collections.Counter(ids)!={i:updates//5 for i in SUPPORT_IDS}:raise ContractError('Unequal or missing source exposures.')
 return dict(updates=updates,per_image=updates//5,counts=dict(collections.Counter(ids)))
def verify_paired_traces(a,b):
 if len(a)!=len(b):raise ContractError('Paired trace lengths differ.')
 for index,(left,right) in enumerate(zip(a,b)):
  if canonical(left)!=canonical(right):raise ContractError('Source, geometry, GT or RNG drift at exposure '+str(index+1))
 return digest_bytes(canonical(a))
def verify_initializations(reports):
 if not reports or len({(r['initial_state_sha256'],r['trainable_names_sha256']) for r in reports})!=1 or not all(r['optimizer_initial_state_empty'] is True for r in reports):raise ContractError('Model, trainable parameter set or fresh optimizer initialization differs.')
def validate_pretrained_key_schema(missing,extra):
 logs={'bbox_head.cls_branches.'+str(i)+'.log_scale' for i in range(7)}
 ignored={'language_model.language_backbone.body.model.embeddings.position_ids','language_model.language_backbone.body.model.pooler.dense.weight','language_model.language_backbone.body.model.pooler.dense.bias'}
 if set(missing)!={'dn_query_generator.label_embedding.weight'}|logs or set(extra)!=ignored:raise ContractError('Pretrained key schema differs from pinned official config/checkpoint.')
 return {'log_scale_keys':sorted(logs),'ignored_checkpoint_keys':sorted(ignored)}
