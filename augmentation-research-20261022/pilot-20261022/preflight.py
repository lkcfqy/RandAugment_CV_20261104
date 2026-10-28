"""Predeclared resource calibration: dummy targets, no research accuracy inspection."""
from common import *
import gc,resource
RUN_START=time.perf_counter()
def guard():
    assert time.perf_counter()-RUN_START<900,'Dummy resource calibration cap900s exceeded.'
    assert not (ROOT/'STOP.request').exists(),'Stop requested before scientific fitting.'

def main():
    setup_mps()
    guard()
    # Prove augmentation touches neither the surrounding CPU nor MPS RNG state.
    probe=Image.fromarray(np.zeros((256,320,3),dtype=np.uint8))
    torch.mps.manual_seed(20371);before_mps=torch.mps.get_rng_state().clone()
    before_cpu=torch.random.get_rng_state().clone()
    for aug in ('weak','ra'):augmented_pil(probe,'rng-isolation',0,20261022,aug)
    assert torch.equal(before_mps,torch.mps.get_rng_state())
    assert torch.equal(before_cpu,torch.random.get_rng_state())
    manifest=json.loads((ROOT/'fit-manifest.json').read_text())
    rows=[r for r in validated_rows(manifest) if r['candidate_partition']=='train']
    images=[];t=time.perf_counter()
    for r in rows:
        guard()
        im=Image.open(ROOT/'data/images'/(r['source_id']+'.jpg')).convert('RGB')
        images.append(im.copy())
    decode=time.perf_counter()-t
    t=time.perf_counter()
    augmented=[]
    for r,im in zip(rows[:32],images[:32]):augmented.append(tensor(augmented_pil(im,r['source_id'],0,20261022,'ra')))
    augmentation_s=time.perf_counter()-t
    receipt={'purpose':'Resource-only calibration with dummy targets, discarded model; no task accuracy or configuration selection.',
        'actual_process_nice':os.getpriority(os.PRIO_PROCESS,0),
        'training_source_count':len(rows),'fit_manifest_sha256':sha(ROOT/'fit-manifest.json'),
        'input_images_decode_seconds':decode,'32_images_RA_tensor_seconds':augmentation_s,
        'batch':16,'cases':[],'MPS_fallback':False,'CPU_threads':1,'MPS_memory_fraction':.1}
    receipt['augmentation_restores_CPU_and_preserves_MPS_RNG']=True
    for layout in ('front','back'):
        guard()
        model=build_model(layout,20261022).to('mps').train();opt=optimizer(model)
        x=torch.stack(augmented[:16]).to('mps')
        # Artificial labels are explicitly independent of actual breed labels.
        y=torch.arange(16,device='mps')%37
        times=[];observed_driver=[]
        for _ in range(5):
            guard()
            torch.mps.synchronize();t=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x),y)
            loss.backward();opt.step();torch.mps.synchronize()
            times.append(time.perf_counter()-t)
            assert torch.isfinite(loss)
            observed_driver.append(torch.mps.driver_allocated_memory())
            time.sleep(2*times[-1])
            guard()
        receipt['cases'].append({'layout':layout,'device':str(x.device),'step_seconds':times,
            'steady_mean_seconds':float(np.mean(times[1:])),
            'allocated_MPS_bytes':torch.mps.current_allocated_memory(),
            'driver_MPS_bytes':torch.mps.driver_allocated_memory(),
            'max_observed_driver_MPS_bytes':max(observed_driver),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        del model,opt,x,y;gc.collect();torch.mps.empty_cache()
    receipt['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['wall_seconds']=time.perf_counter()-RUN_START
    receipt['status']='completed_resource_only_calibration'
    receipt['estimated_30ep_four_cell_seconds_compute_plus_pacing']=sum(r['steady_mean_seconds'] for r in receipt['cases'])*2*math.ceil(len(rows)/16)*30*3
    receipt['estimate_excludes']='Evaluation, CPU loading/augmentation and checkpoint I/O; measure actual completed times separately.'
    guard();dump(ROOT/'preflight.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
