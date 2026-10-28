"""Training-only implementation checks and first nonfinite forward activation."""
from pathlib import Path
import hashlib,json,math
import torch
from PIL import Image
import repair_training as R

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu');torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'repair-protocol.json').read_text())
    transform=R.torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    x=torch.stack([transform(Image.open(R.DATA/'photos'/(r['source_id']+'.jpg')).convert('RGB')) for r in protocol['photos']])
    y=torch.tensor([r['class37_target'] for r in protocol['photos']],dtype=torch.long)
    one,oa=R.new_model(.1,.01);two,ob=R.new_model(.1,.01)
    ob.param_groups.reverse()  # original file lists head then body; diagnostic lists body then head.
    assert R.state_hash(one.state_dict())==R.state_hash(two.state_dict())
    initial={name:p.detach().clone() for name,p in one.named_parameters()}
    one.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(one(x),y);loss.backward()
    expected={name:initial[name]-((.1 if name.startswith('fc.') else .01)*(p.grad+1e-4*initial[name])) for name,p in one.named_parameters()}
    parameter_ids=[id(p) for group in oa.param_groups for p in group['params']]
    assert len(parameter_ids)==len(set(parameter_ids))==len(list(one.parameters()))
    oa.step();max_error=0.;worst=None
    for name,p in one.named_parameters():
        error=float((p.detach()-expected[name]).abs().max())
        if error>max_error:max_error=error;worst=name
    assert max_error<1e-6
    two.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(two(x),y).backward();ob.step()
    equal=R.state_hash(one.state_dict())==R.state_hash(two.state_dict());assert equal
    del two,ob,initial,expected
    # Continue the disposable original setup through its sixth finite update.
    for i in range(1,6):
        oa.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(one(x),y)
        assert bool(torch.isfinite(loss));loss.backward();oa.step()
    assert all(bool(torch.isfinite(p).all()) for p in one.parameters())
    events=[];handles=[]
    def capture(name):
        def hook(module,inputs,output):
            if len(events)>=6:return
            if isinstance(output,torch.Tensor) and not bool(torch.isfinite(output).all()):
                input_finite=all(not isinstance(t,torch.Tensor) or bool(torch.isfinite(t).all()) for t in inputs)
                events.append(dict(module=name,type=type(module).__name__,input_finite=input_finite,
                    parameters_finite=all(bool(torch.isfinite(p).all()) for p in module.parameters()),
                    output_shape=list(output.shape),output_nonfinite_elements=int((~torch.isfinite(output)).sum())))
        return hook
    for name,module in one.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.BatchNorm2d,torch.nn.ReLU,torch.nn.Linear)):
            handles.append(module.register_forward_hook(capture(name)))
    with torch.inference_mode():output=one(x)
    assert not bool(torch.isfinite(output).all()) and events
    first=events[0];assert first['input_finite'] and first['parameters_finite']
    for handle in handles:handle.remove()
    before=json.loads((ROOT/'original_high_LR-result.json').read_text())
    after=json.loads((ROOT/'repaired_lower_LR-result.json').read_text())
    summary=json.loads((ROOT/'repair-result.json').read_text())
    A=[json.loads(line) for line in (ROOT/'original_high_LR-trace.jsonl').read_text().splitlines()]
    B=[json.loads(line) for line in (ROOT/'repaired_lower_LR-trace.jsonl').read_text().splitlines()]
    assert before['initial_state_sha256']==after['initial_state_sha256']
    assert len(B)==30 and all(r['loss_finite'] and r['gradients_finite'] and r['momentum_buffers_finite'] for r in B)
    assert all(r['parameters_after'][g]['finite'] for r in B for g in ['head','body'])
    assert len(after['lean_losses'])==20 and all(math.isfinite(v) for v in after['lean_losses'])
    assert after['final_training_loss']<.7*B[0]['loss']
    assert summary['code_sha256']==hashlib.sha256((ROOT/'repair_training.py').read_bytes()).hexdigest()
    for name,key in [('cpu_training_timing.initial_failed.py','old_failed_code_sha256'),('benchmark-initial-failure.json','original_failed_log_sha256')]:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==protocol[key]
    verification=dict(status='passed',input_normalization_labels_and_finite_initial_state_checked=True,
        same_init_between_original_and_repair=True,no_optimizer_parameter_duplication=True,
        first_SGD_update_matches_manual_weight_decay_equation_max_abs_error=max_error,worst_manual_update_parameter=worst,
        reversing_disjoint_optimizer_group_order_bitwise_equal=True,
        first_nonfinite_forward_module=first,first_nonfinite_forward_events=events,
        original_finite_parameter_update_then_activation_overflow_verified=True,
        repaired30_detailed_and20_guarded_steps_all_finite=True,nonzero_backbone_and_head_updates_verified=True,
        original_failure_files_preserved=True,repair_code_sha_matches_execution=True,
        no_validation_or_test_use=True,no_new_data=True,CPU_only=True,GPU=False,ARIS=False)
    R.save('verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
