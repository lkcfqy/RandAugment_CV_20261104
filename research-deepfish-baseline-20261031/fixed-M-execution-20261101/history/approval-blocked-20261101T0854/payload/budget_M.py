"""Frozen whole-allocation25% cost formula. Pure arithmetic; no runtime effects."""
from contracts import ContractError, positive, MAX_CU, MAX_GPU_SECONDS

REFERENCE_UPDATE_SECONDS=1.6807091929999842
REFERENCE_INIT_SECONDS=20.244484609999972
REFERENCE_BPS=4453855.0958829
PREPARATION_SECONDS=554.525631070137
CHECKPOINT_BYTES=692175962

def gate(ctx,elapsed,*,updates_remaining=200,init_remaining=REFERENCE_INIT_SECONDS,
         t_update=REFERENCE_UPDATE_SECONDS,bps=REFERENCE_BPS,prepare_remaining=0.,
         checkpoint_bytes=CHECKPOINT_BYTES,receipt_remaining=120.):
 rate=ctx['cu_per_hour_upper_bound'];prior=ctx['prior_cu'];prior_gpu=ctx['prior_gpu_seconds']
 if not all(positive(x) for x in (rate,bps,t_update,checkpoint_bytes)):
  raise ContractError('Finite positive rate/throughput/full-checkpoint quantities required.')
 if elapsed<0 or init_remaining<0 or prepare_remaining<0 or receipt_remaining<0 or not 0<=updates_remaining<=200:
  raise ContractError('Invalid remaining complete-run work.')
 if prior<1.8088330815795528 or prior_gpu<5443.899788473129:
  raise ContractError('Historical ledger cannot be reset.')
 t=max(REFERENCE_UPDATE_SECONDS,t_update);B=min(REFERENCE_BPS,bps)
 available=min((MAX_CU-prior)*3600/rate,MAX_GPU_SECONDS-prior_gpu)
 remaining=prepare_remaining+init_remaining+1.5*updates_remaining*t+2*checkpoint_bytes/B+receipt_remaining+180.
 total=1.25*(elapsed+remaining)
 return dict(status='FITS' if total<=available else 'RESOURCE_UNDETERMINED',
   elapsed_allocation_seconds=elapsed,required_total_seconds=total,available_total_seconds=available,
   required_remaining_seconds=total-elapsed,remaining_seconds=available-elapsed,
   projected_cumulative_CU=prior+total*rate/3600,projected_cumulative_GPU_seconds=prior_gpu+total,
   whole_margin=1.25,release_seconds_before_margin=180.,updates_remaining=updates_remaining,
   init_remaining=init_remaining,prepare_remaining=prepare_remaining,t_update_bound=t,B_bound=B,
   checkpoint_bytes=checkpoint_bytes,receipt_remaining=receipt_remaining,
   required_seconds=total-elapsed)
