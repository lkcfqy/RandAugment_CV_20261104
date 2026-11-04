"""Single64-dev CPU schedule, permitted only after verified full return and GPU release."""
import copy,json,os,pathlib,subprocess,sys,time
from diagnostic_contracts import P,R,PREP,sha,atomic_json,complete_matrix

def prepare():
 base=P.parent;host=base/'gpu-run';control=json.loads((host/'controller_report.json').read_text())
 usage=control['final_sessions_usage']
 if control['status']!='COMPLETE' or control['release']!='CONFIRMED' or usage['active_assignments'] or usage['assignments'] or usage['rate_per_hour']!=0:
  raise RuntimeError('GPU completion and independently reconciled release required before CPU model execution')
 receipt=host/'LOCAL_M_VERIFIED.json';verified=json.loads(receipt.read_text())
 if verified['status']!='LOCAL_M_CHECKPOINT_VERIFIED' or verified['actual_updates']!=200:raise RuntimeError('Full readable U200 checkpoint receipt required')
 original=R/'cpu-cross-input-diagnostic-20261101';old=json.loads((original/'RUN_PROTOCOL.json').read_text())
 checkpoint=host/'artifacts'/verified['checkpoint']['path'];config=base/'payload/config_A_reference.json'
 if sha(checkpoint)!=verified['checkpoint']['sha256']:raise RuntimeError('Full returned M checkpoint changed')
 if sha(config)!=old['models'][0]['config_sha256']:raise RuntimeError('Original A inference configuration changed')
 spec=dict(seed=20261031,arm='M',checkpoint=str(checkpoint),bytes=checkpoint.stat().st_size,
           sha256=verified['checkpoint']['sha256'],config=str(config),config_sha256=sha(config),state_receipt=str(receipt))
 run={k:copy.deepcopy(old[k]) for k in ('original_protocol_sha256','environment_lock_sha256','core_runtime_versions',
       'dev_annotation','image_order','image_manifest','runtime','metric_mapping','official_implementation_sha256') if k in old}
 run.update(authorized_fixed_M_CPU64=True,models=[spec],expected_identities=[],identity_map={},
    reference_A_metrics=str(original/'metrics/20261031_A_c100.json'),code_sha256={n:sha(P/n) for n in ('formal_cpu.py','diagnostic_contracts.py','run_cpu.py')},
    primary_metric='AP@100',official_test=False,training=False,GPU_MPS_Colab_upload_CU_new=0)
 for image_id in run['image_order']:
  old_key=f'20261031_A_c100_{image_id}';key=f'20261031_M_c100_{image_id}'
  record=json.loads((original/'predictions'/(old_key+'.json')).read_text())['payload']
  pinned=copy.deepcopy(old['identity_map'][old_key]);pinned.update(arm='M',checkpoint_sha256=spec['sha256'],config_sha256=spec['config_sha256'],reference_A_geometry_sha256=record['geometry_sha256'])
  run['expected_identities'].append(key);run['identity_map'][key]=pinned
 for name in ('predictions','receipts','metrics'):(P/name).mkdir(exist_ok=False)
 atomic_json(P/'RUN_PROTOCOL.json',run,new=True)
 atomic_json(P/'RUN_STARTED.json',dict(start_monotonic=time.monotonic(),deadline_monotonic=time.monotonic()+900,wall_limit_seconds=900,GPU_release_verified=True),new=True)
 return run

def main():
 start=time.monotonic();run=prepare();phases=[]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(P/'mplconfig'))
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
 try:
  for phase in ('infer','metrics','verify'):
   t=time.monotonic()
   with (P/(phase+'.log')).open('xb') as log:
    result=subprocess.run([sys.executable,'-B',str(P/'formal_cpu.py'),'--phase',phase],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,900-(time.monotonic()-start)))
   phases.append(dict(phase=phase,returncode=result.returncode,seconds=time.monotonic()-t))
   if result.returncode:raise RuntimeError('CPU phase failed: '+phase)
   if phase=='infer':atomic_json(P/'ALL_64_COMPLETE.json',complete_matrix(run),new=True)
   print(json.dumps(phases[-1]),flush=True)
  status='COMPLETE_CPU64_RECOMPUTED'
 except BaseException as error:
  atomic_json(P/'STATUS.json',dict(status='INCOMPLETE',error_type=type(error).__name__,reason=str(error),phases=phases,seconds=time.monotonic()-start),new=True)
  raise
 atomic_json(P/'STATUS.json',dict(status=status,phases=phases,seconds=time.monotonic()-start,all_CPU_processes_exited=True,GPU_MPS_upload_CU_new=0),new=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
