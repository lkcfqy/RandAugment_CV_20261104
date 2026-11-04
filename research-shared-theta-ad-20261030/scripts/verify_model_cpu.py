"""Offline CPU structural smoke check with one synthetic tensor; never reads dataset images."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.metadata,inspect,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
START=time.monotonic()
import torch
from safetensors import safe_open
from transformers import AutoModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
assert torch.get_default_device().type=='cpu'
torch.set_num_threads(4)
model_dir=ROOT/'models/dinov2-small'
weight=model_dir/'model.safetensors'
sha=hashlib.sha256(weight.read_bytes()).hexdigest()
assert sha=='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
with safe_open(str(weight),framework='pt',device='cpu') as f:
    tensor_shapes={k:list(f.get_slice(k).get_shape()) for k in f.keys()}
    tensor_metadata=f.metadata()
(ROOT/'reports/safetensors-structure.json').write_text(json.dumps({'metadata':tensor_metadata,'shapes':tensor_shapes},indent=2))
# Reject any accidental legacy loader path. No globals or allowlists are registered.
def no_pickle(*args,**kwargs):raise RuntimeError('torch.load is forbidden in this asset check')
torch.load=no_pickle
model,loading=AutoModel.from_pretrained(str(model_dir),trust_remote_code=False,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,output_loading_info=True)
model=model.to('cpu').eval().requires_grad_(False)
assert isinstance(model,Dinov2Model)
assert model.config.hidden_size==384 and model.config.patch_size==14
assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'] and not loading['error_msgs']
assert {p.device.type for p in model.parameters()}=={'cpu'}
# Synthetic data is solely a geometry and final-layernorm contract check.
with torch.inference_mode():
    tic=time.monotonic()
    result=model(pixel_values=torch.zeros(1,3,224,224,device='cpu'),output_hidden_states=True,return_dict=True)
    forward_seconds=time.monotonic()-tic
    norm=model.layernorm(result.hidden_states[-1])
    max_abs=float((norm-result.last_hidden_state).abs().max())
    assert max_abs==0.0
    assert list(result.last_hidden_state.shape)==[1,257,384]
    patches=result.last_hidden_state[:,1:,:].reshape(1,16,16,384)
    assert torch.isfinite(patches).all()
source=inspect.getsource(Dinov2Model.forward)
(ROOT/'sources/transformers-dinov2-forward.py.txt').write_text(source)
packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','safetensors','Pillow','numpy','huggingface_hub')}
report={'status':'passed','model':'facebook/dinov2-small','revision':json.loads((ROOT/'sources/hf-model-info.json').read_text())['sha'],'weights_sha256':sha,'weight_bytes':weight.stat().st_size,'parameter_count':sum(p.numel() for p in model.parameters()),'safe_format':'safetensors','trust_remote_code':False,'local_files_only':True,'torch_load_forbidden_during_check':True,'device':'cpu','gpu_used':False,'real_image_model_forwards':0,'synthetic_input_shape':[1,3,224,224],'last_hidden_state_shape':list(result.last_hidden_state.shape),'patch_grid_shape':list(patches.shape),'patch_size':14,'hidden_dim':384,'final_norm_attribute':'model.layernorm','final_norm_output':'outputs.last_hidden_state[:, 1:, :]','final_norm_max_abs_difference':max_abs,'norm_epsilon':model.layernorm.eps,'loading_info':loading,'synthetic_forward_seconds':forward_seconds,'wall_seconds':time.monotonic()-START,'packages':packages,'python':sys.version,'executable':sys.executable,'forward_source_sha256':hashlib.sha256(source.encode()).hexdigest()}
(ROOT/'reports/model-cpu-verification.json').write_text(json.dumps(report,indent=2))
contract={'status':'geometry requirement fixed; complete scientific preprocessing protocol pending independent review','image_input':'decoded RGB','geometry':{'resize':'whole image directly to 224x224 square','center_crop':False,'short_edge_resize':False,'patch_grid':[16,16]},'official_default_not_used':{'shortest_edge':256,'center_crop':[224,224]},'normalization_reference':{'rescale':1/255,'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225]},'still_to_predeclare':['interpolation and antialias implementation','brightness/contrast definition, order, levels and clipping','feature L2 normalization and nearest-neighbor distance','shared transformation selection and all evaluation decisions'],'test_data_used_for_model_check':False}
(ROOT/'reports/preprocessing-contract.json').write_text(json.dumps(contract,indent=2))
print(json.dumps(report,indent=2))
