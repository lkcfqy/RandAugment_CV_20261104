"""Compare actual CPU/MPS logs against pre-fixed numerical and speed gates."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text())
def main():
    protocol=load('protocol.json');cpu=load('cpu-result.json');mps=load('mps-result.json')
    assert cpu['code_sha256']==mps['code_sha256']==hashlib.sha256((ROOT/'worker.py').read_bytes()).hexdigest()
    assert cpu['weight_sha256']==mps['weight_sha256'] and cpu['threads']==mps['threads']==1
    limits=protocol['numerical_acceptance'];rows=[]
    for c,g in zip(cpu['results'],mps['results']):
        assert c['batch']==g['batch'] and c['inference']['initial_state_sha256']==g['inference']['initial_state_sha256']
        assert c['training']['initial_state_sha256']==g['training']['initial_state_sha256']
        x=np.array(c['inference']['logits']);z=np.array(g['inference']['logits']);diff=z-x
        absmax=float(np.max(np.abs(diff)));relative=float(np.linalg.norm(diff)/max(np.linalg.norm(x),1e-12))
        agreement=float(np.mean(x.argmax(axis=1)==z.argmax(axis=1)))
        cx=np.array(c['training']['logits_after10updates']);gz=np.array(g['training']['logits_after10updates'])
        training_agreement=float(np.mean(cx.argmax(axis=1)==gz.argmax(axis=1)))
        lossdiff=float(np.max(np.abs(np.array(c['training']['losses'])-np.array(g['training']['losses']))))
        gates=dict(inference_abs_error=absmax<=limits['inference_max_abs_logit_error'],
            inference_relative_error=relative<=limits['inference_relative_L2_error'],
            inference_argmax=agreement==1.,training_finite=c['training']['all_loss_grad_parameter_finite'] and g['training']['all_loss_grad_parameter_finite'],
            training_loss_decline=c['training']['training_loss_decreased'] and g['training']['training_loss_decreased'],
            training_loss_trajectory_error=lossdiff<=limits['max_absolute_per_step_training_loss_difference'],
            training_argmax=training_agreement==1.)
        speed=g['training']['images_per_second']/c['training']['images_per_second']
        rows.append(dict(batch=c['batch'],CPU_inference_images_sec=c['inference']['images_per_second'],
            MPS_inference_images_sec=g['inference']['images_per_second'],inference_speedup=g['inference']['images_per_second']/c['inference']['images_per_second'],
            CPU_guarded_training_images_sec=c['training']['images_per_second'],MPS_guarded_training_images_sec=g['training']['images_per_second'],
            guarded_training_speedup=speed,inference_max_abs_logit_error=absmax,inference_relative_L2_error=relative,
            inference_argmax_agreement=agreement,max_short_training_loss_trajectory_abs_error=lossdiff,
            short_training_argmax_agreement=training_agreement,CPU_initial_final_loss=[c['training']['losses'][0],c['training']['final_loss']],
            MPS_initial_final_loss=[g['training']['losses'][0],g['training']['final_loss']],numerical_gates=gates,
            all_numerical_gates_passed=all(gates.values()),useful_measured_speedup=speed>=1.2,
            accepted_for_future_similar_local_development=all(gates.values()) and speed>=1.2,
            MPS_max_sampled_tensor_allocated_bytes=max(r['current_allocated_memory_bytes'] for r in g['training']['memory_samples_after_steps']),
            MPS_max_sampled_driver_allocated_bytes=max(r['driver_allocated_memory_bytes'] for r in g['training']['memory_samples_after_steps']),
            MPS_inference_warmup_seconds=g['inference']['warmup_seconds'],MPS_training_warmup_seconds=g['training']['warmup_seconds']))
    result=dict(status='completed_measured_CPU_MPS_comparison',rows=rows,
        CPU_process_peak_RSS_bytes=cpu['memory_final']['process_peak_RSS_bytes_macos'],
        MPS_process_peak_RSS_bytes=mps['memory_final']['process_peak_RSS_bytes_macos'],
        CPU_worker_wall_seconds=cpu['wall_seconds'],MPS_worker_wall_seconds=mps['wall_seconds'],
        memory_caution='MPS tensor/driver values are samples, not guaranteed peaks; Apple unified memory overlaps process/RAM accounting, do notadd them.',
        scope='same8training photos and fixedFP32model only; not a whole-pilot or allworkload speed benchmark; oldCPUresults unchanged',
        A100_or_remote_GPU=False,new_install_or_security_change=False,no_test_use=True,ARIS=False)
    with (ROOT/'comparison.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
