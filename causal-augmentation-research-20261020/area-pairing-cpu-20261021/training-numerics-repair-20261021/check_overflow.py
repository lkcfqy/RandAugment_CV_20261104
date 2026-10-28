"""Compute a few failed FP32 convolution entries in FP64 using the same training fixture."""
from pathlib import Path
import json,math
import torch
from PIL import Image
import repair_training as R

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'repair-protocol.json').read_text())
    transform=R.torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    x=torch.stack([transform(Image.open(R.DATA/'photos'/(r['source_id']+'.jpg')).convert('RGB')) for r in protocol['photos']])
    y=torch.tensor([r['class37_target'] for r in protocol['photos']],dtype=torch.long)
    model,optimizer=R.new_model(.1,.01)
    for i in range(6):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x),y)
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    evidence=[];limit=torch.finfo(torch.float32).max
    def hook(module,inputs,output):
        bad=torch.nonzero(~torch.isfinite(output))[:3]
        padded=torch.nn.functional.pad(inputs[0],(1,1,1,1))
        for entry in bad:
            n,c,oy,ox=[int(v) for v in entry]
            patch=padded[n,:,oy:oy+3,ox:ox+3].double()
            weight=module.weight[c].detach().double()
            products=patch*weight;exact=float(products.sum())
            evidence.append(dict(index=[n,c,oy,ox],FP32_output=R.number(output[n,c,oy,ox]),
                FP64_dot_product=exact,FP32_max_finite=limit,
                FP64_magnitude_exceeds_FP32_range=abs(exact)>limit,
                maximum_abs_FP64_product=float(products.abs().max()),FP64_sum_abs_products=float(products.abs().sum())))
    layer=model.layer2[0].conv2
    assert layer.kernel_size==(3,3) and layer.stride==(1,1) and layer.padding==(1,1) and layer.bias is None
    handle=layer.register_forward_hook(hook)
    with torch.inference_mode():model(x)
    handle.remove();assert evidence and all(math.isfinite(r['FP64_dot_product']) for r in evidence)
    R.save('fp32-overflow-evidence.json',dict(status='failed_convolution_coordinates_recomputed_in_FP64',
        module='layer2.0.conv2',all_inputs_and_parameters_finite=True,evidence=evidence,
        supports_real_output_out_of_FP32_range=any(r['FP64_magnitude_exceeds_FP32_range'] for r in evidence),
        CPU_only=True,no_test_use=True,no_new_data=True))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
