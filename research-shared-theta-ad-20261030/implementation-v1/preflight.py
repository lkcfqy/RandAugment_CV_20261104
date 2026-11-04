"""At most one 5-minute MPS window, source-normal only, timing and numeric facts only."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['PYTORCH_MPS_FAST_MATH']='0'
import json,math,threading,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from stheta.protocol import ASSETS,WORK,SEEDS,SPEC,file_sha,save_json
from stheta.data import read_rgb
from stheta.encoder import load_encoder,encode,color_normalize
from stheta.methods import distance_cache,all_readouts,upper_mean
from stheta.runtime import ComputeWindow,prepare_seed,sync,disk_bytes,save_query_cache,save_array,verify_frozen

def timed(fn):
    sync('mps');start=time.perf_counter();value=fn();sync('mps')
    return value,time.perf_counter()-start

@torch.inference_mode()
def mps_numeric_check(model,window):
    torch.manual_seed(5501)
    q=F.normalize(torch.randn(7,48),dim=-1);bank=F.normalize(torch.randn(3,9,16,48),dim=-1)
    u,_=torch.linalg.qr(torch.randn(48,32));bases={'S0':u.numpy()}
    cpu=distance_cache(q,bank,bases,group_chunk=4)
    gpu=distance_cache(q.to('mps'),bank.to('mps'),bases,group_chunk=4,deadline_check=window.check)
    maxabs=float(np.abs(cpu[0]-gpu[0]).max());assert maxabs<=1e-4
    np.testing.assert_array_equal(cpu[1],gpu[1]);projection_error=0.
    for key in cpu[2]:
        for mode in cpu[2][key]:
            a,b=cpu[2][key][mode],gpu[2][key][mode]
            projection_error=max(projection_error,float(np.abs(a['distances']-b['distances']).max()))
            np.testing.assert_array_equal(a['indices'],b['indices'])
    assert projection_error<=1e-4
    fixture=np.load(WORK/'reports/encoder-fixed-input.npy',allow_pickle=False)
    expected=np.load(WORK/'reports/encoder-current457.npz',allow_pickle=False)['normalized_patch']
    result=model(pixel_values=torch.from_numpy(fixture).to('mps'),return_dict=True).last_hidden_state[:,1:,:]
    unit=F.normalize(result,dim=-1,eps=1e-8).cpu().numpy();diff=np.abs(unit-expected)
    assert np.isfinite(unit).all() and float(diff.max())<=1e-4 and float(diff.mean())<=1e-5
    return {'status':'passed','data':'synthetic only','cache_maxabs_cpu_mps':maxabs,'projection_maxabs_cpu_mps':projection_error,
      'normalized_encoder_patch_maxabs_cpu_mps':float(diff.max()),'normalized_encoder_patch_meanabs_cpu_mps':float(diff.mean()),
      'fixed_tolerances':{'distance_maxabs':1e-4,'normalized_patch_maxabs':1e-4,'normalized_patch_meanabs':1e-5}}

def main():
    frozen=verify_frozen();manifest=json.loads((WORK/'split-manifest.json').read_text())
    assert manifest['status']=='ready' and frozen['formal_scoring_authorized'] is False
    assert json.loads((WORK/'reports/encoder-cross-version.json').read_text())['status']=='passed'
    if not torch.backends.mps.is_available():
        save_json(WORK/'reports/throughput-preflight.json',{'status':'blocked','reason':'MPS unavailable; no GPU allocation elsewhere','formal_scoring_run':False});return
    torch.set_num_threads(4);model=load_encoder();window=ComputeWindow(preflight=True)
    log={'status':'running','scientific_metrics_computed':False,'real_theta_distributions_computed':False,'test_images_read_by_preflight':0,
         'frozen_manifest_sha256':file_sha(WORK/'frozen-manifest.json'),'method_count':59,'external_baseline_count':51}
    state=None;query=None;cq=None;disk_check_seconds=0.
    def timeout():
        log.update({'status':'stopped_at_preflight_wall_limit','elapsed_seconds':window.elapsed(),'reason':'5-minute budget watchdog; no full evaluation started'})
        save_json(WORK/'reports/throughput-preflight.json',log);os._exit(124)
    window.begin();watchdog=threading.Timer(max(0,295-window.elapsed()),timeout);watchdog.daemon=True;watchdog.start()
    try:
        model.to('mps');log['MPS_numeric_check']=mps_numeric_check(model,window)
        print('Synthetic CPU/MPS checks passed. Building seed11 source-normal bank.',flush=True)
        with threadpool_limits(limits=4):
            state=prepare_seed(model,manifest,11,'mps',window,WORK/'preflight-cache'/'seed11')
            log['setup_timings']=state['timings'];log['reference_artifacts']=state['artifacts']
            query_path=manifest['calibration'][0]['canonical']['path']
            assert '/train/good/' in query_path and all('/train/good/' in p for p in state['reference_source_paths'])
            log['fixed_query_path']=query_path;log['ref_cluster_ids']=state['ref_ids'];log['actual_bank_shape']=list(state['bank'].shape)
            start=time.perf_counter();rgb=read_rgb(ASSETS/'data'/query_path);decode_seconds=time.perf_counter()-start
            q_cpu,encode_seconds=timed(lambda:encode(model,[rgb],'mps',batch_size=1,deadline_check=window.check))
            query=q_cpu[0].to('mps')
            start=time.perf_counter();normalized,floor=color_normalize(rgb,state['target']);color_prepare_seconds=time.perf_counter()-start
            cq_cpu,color_encode_seconds=timed(lambda:encode(model,[normalized],'mps',batch_size=1,deadline_check=window.check));cq=cq_cpu[0].to('mps')
            samples=[]
            for repeat in range(3):
                window.check(reserve=10)
                (A,argq,projected),dist_time=timed(lambda:distance_cache(query,state['bank'],state['bases'],group_chunk=32,deadline_check=window.check))
                scores,readout_time=timed(lambda:all_readouts(query,state['bank'],A,argq,projected,state['bases'],state['ref_ids'],state['centers'],state['ref_hist']))
                (cA,cix,_),color_dist_time=timed(lambda:distance_cache(cq,state['color_bank'],group_chunk=32,deadline_check=window.check))
                scores['robust_color']=float(upper_mean(cA[:,:,0].min(1)))
                assert len(scores)==59 and np.isfinite(list(scores.values())).all()
                # Source-normal scalar scores are transient and discarded; no method scores or theta choices are persisted.
                del scores
                start=time.perf_counter()
                cache=save_query_cache(WORK/'preflight-cache'/f'source-query-{repeat}.npz',A,argq,projected,cA,cix)
                feature=save_array(WORK/'preflight-cache'/f'source-feature-{repeat}.npy',query.cpu().numpy())
                _=np.load(WORK/feature['path'],allow_pickle=False)
                io_time=time.perf_counter()-start
                if repeat==0:
                    start=time.perf_counter();window.check_disk();disk_check_seconds=time.perf_counter()-start
                samples.append({'exact_distance_all_pre_projection_seconds':dist_time,'selectors_bow_post_projection_seconds':readout_time,
                  'robust_color_distance_seconds':color_dist_time,'persistence_hash_read_seconds':io_time,'cache_bytes':cache['bytes'],'feature_bytes':feature['bytes'],
                  'cache_sha256':cache['sha256'],'feature_sha256':feature['sha256']})
                print(json.dumps({'repeat':repeat+1,'distance_seconds':dist_time,'readout_seconds':readout_time,'elapsed_window_seconds':window.elapsed()}),flush=True)
            log['samples']=samples
            st=state['timings'];per_forward=max(encode_seconds,color_encode_seconds,st['ref_encode_576']/576,st['color_ref_encode_64']/64)
            read_per_image=max(decode_seconds,st['ref_decode']/64)
            normalize_per_image=max(color_prepare_seconds,st['color_ref_prepare']/64)
            maxsample=lambda k:max(s[k] for s in samples)
            components={
              'encoder_6024_forwards':6024*per_forward,
              'decode_3270_images':3270*read_per_image,
              'reference_photometric_views_three_seeds':3*st['ref_view_prepare'],
              'color_normalize_3270_images':3270*normalize_per_image,
              'color_target_projection_bow_fits_three_seeds':3*(st['color_target_fit']+st['projection_fit']+st['bow_fit_and_reference_hist']),
              'setup_persistence_three_seeds':3*st['setup_persistence'],
              'distance_pre_projection_3078_queries':3078*maxsample('exact_distance_all_pre_projection_seconds'),
              'selectors_bow_post_projection_3078_queries':3078*maxsample('selectors_bow_post_projection_seconds'),
              'color_distance_3078_queries':3078*maxsample('robust_color_distance_seconds'),
              'persistence_hash_read_3078_queries':3078*maxsample('persistence_hash_read_seconds'),
              'disk_guard_every16_queries':math.ceil(3078/16)*disk_check_seconds,
              'CPU_final_calibration_metrics_and_report_allowance':30.}
            disk_now=disk_bytes()
            # Keep all three seed banks/caches, original query features once, source artifacts and reference environment.
            growth=3*sum(x['bytes'] for x in state['artifacts'])+3078*maxsample('cache_bytes')+1026*maxsample('feature_bytes')+128*1024**2
            remaining_plain=sum(components.values());elapsed=window.elapsed();total=elapsed+1.2*remaining_plain
            log.update({'status':'complete','all_methods_included':True,'query_count_per_seed':1026,'seed_count':3,'forward_count_upper_bound':6024,
              'per_forward_seconds_conservative':per_forward,'forecast_components_seconds':components,'remaining_full_run_plain_seconds':remaining_plain,
              'forecast_margin':1.2,'includes_already_elapsed_wall':True,'elapsed_at_forecast_seconds':elapsed,'forecast_total_from_first_GPU_seconds':total,
              'full_protocol_fits_120min':total<=7200,'allocated_disk_now_bytes':disk_now,'forecast_additional_retained_bytes':growth,
              'forecast_peak_disk_bytes':disk_now+growth,'fits_8GiB':disk_now+growth<=8*1024**3,
              'first_gpu_epoch_seconds':window.start,'absolute_total_deadline_epoch_seconds':window.start+7200,
              'forecast_note':'Complete protocol rebuilt from scratch; max measured sample and conservative per-forward rate, then 20 percent margin; future review/CPU waiting continues consuming the same absolute wall window',
              'model_forwards_on_real_test_images':0,'colab_allocations':0,'formal_scoring_run':False,
              'preflight_source_normal_forwards':576+64+1+1,'model_synthetic_MPS_inputs':3,'source_images_or_labels_used_for_hyperparameter_selection':False})
            log['full_protocol_fits_120min']=log['full_protocol_fits_120min'] and log['fits_8GiB']
    except Exception as err:
        log.update({'status':'failed_or_budget_stopped','error':str(err),'traceback':traceback.format_exc(),'formal_scoring_run':False,'elapsed_seconds':window.elapsed()})
        raise
    finally:
        if state is not None:del state
        del query,cq
        model.to('cpu');sync('mps');torch.mps.empty_cache();sync('mps')
        log['completed_window_wall_seconds']=window.elapsed();log['GPU_released']=True
        save_json(WORK/'reports/throughput-preflight.json',log);watchdog.cancel()
    print(json.dumps({k:log[k] for k in ('status','completed_window_wall_seconds','forecast_total_from_first_GPU_seconds','full_protocol_fits_120min','forecast_peak_disk_bytes','fits_8GiB','formal_scoring_run')}),flush=True)

if __name__=='__main__':main()
