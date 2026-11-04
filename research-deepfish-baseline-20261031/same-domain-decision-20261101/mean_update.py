"""One exact source-mean objective per update; no model-specific behavior."""
import math,time

def source_mean_update(model, optim, batches, k, before_step=None, schedule_boundary=None):
    if not isinstance(k,int) or k<1: raise ValueError('Positive frozen source count required')
    if optim._accumulative_counts!=1: raise ValueError('No nested accumulation supported')
    rows=[];start=time.monotonic()
    # Previous update ends with zero_grad, exactly as official update_params.
    for n,batch in enumerate(batches,1):
        if n>k: raise ValueError('Too many source batches')
        t=time.monotonic()
        with optim.optim_context(model):
            data=model.data_preprocessor(batch,True)
            losses=model._run_forward(data,mode='loss')
            loss,log=model.parse_losses(losses)
        if loss.numel()!=1 or not math.isfinite(float(loss.detach())): raise ValueError('Invalid scalar source loss')
        value=float(loss.detach());forward_seconds=time.monotonic()-t;t=time.monotonic()
        optim.backward(loss/k)
        rows.append({'source':n,'unweighted_loss':value,'forward_loss_seconds':forward_seconds,'backward_seconds':time.monotonic()-t})
        del loss,log,losses,data,batch
    if len(rows)!=k: raise ValueError('Missing source batches')
    if before_step is not None: before_step()
    t=time.monotonic();optim.step();optim.zero_grad();step_seconds=time.monotonic()-t
    if schedule_boundary is not None: schedule_boundary()
    return {'source_count':k,'mean_loss':sum(x['unweighted_loss'] for x in rows)/k,'sources':rows,'optimizer_zero_seconds':step_seconds,'update_seconds':time.monotonic()-start}
