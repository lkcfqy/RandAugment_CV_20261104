"""Run installed official built-in Dinov2Model on a locked synthetic 224 fixture."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,inspect,json,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import Dinov2Model
from stheta.protocol import ASSETS,WORK,WEIGHT_SHA,save_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);args=p.parse_args()
    lock=json.loads((WORK/'reports/encoder-comparison-lock.json').read_text())
    input_path=WORK/'reports/encoder-fixed-input.npy';assert file_sha(input_path)==lock['input_sha256']
    assert file_sha(ASSETS/'models/dinov2-small/model.safetensors')==WEIGHT_SHA
    torch.set_num_threads(4)
    def reject(*args,**kwargs):raise RuntimeError('Legacy pickle loader forbidden')
    torch.load=reject
    model,info=Dinov2Model.from_pretrained(str(ASSETS/'models/dinov2-small'),trust_remote_code=False,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    assert all(not info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.to('cpu').eval().requires_grad_(False)
    assert model.config._attn_implementation=='eager'
    x=torch.from_numpy(np.load(input_path,allow_pickle=False)).float()
    with torch.inference_mode():
        out=model(pixel_values=x,output_hidden_states=True,output_attentions=True,return_dict=True)
        last=out.last_hidden_state
        pos=model.embeddings.interpolate_pos_encoding(torch.empty(x.shape[0],257,384),224,224)
        expected=model.layernorm(out.hidden_states[-1]);within_norm=float((last-expected).abs().max())
        patch=last[:,1:,:];unit=F.normalize(patch,p=2,dim=-1,eps=1e-8)
        arrays={'native_position':pos.numpy(),'first_eager_attention':out.attentions[0].numpy(),'last_hidden_state':last.numpy(),'patch_after_cls_drop':patch.numpy(),'normalized_patch':unit.numpy()}
        assert within_norm==0
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert arrays['normalized_patch'].shape==(3,256,384)
    target=WORK/'reports'/f'encoder-{args.label}.npz';np.savez(target,**arrays)
    metadata={'label':args.label,'transformers':transformers.__version__,'torch':torch.__version__,'numpy':np.__version__,
       'device':'cpu','attention':'eager','dtype':'float32','input_sha256':lock['input_sha256'],'weight_sha256':WEIGHT_SHA,
       'source_kind':'installed official PyPI wheel built-in Dinov2Model; no downloaded-source execution',
       'model_source_path':inspect.getfile(Dinov2Model),'model_source_sha256':file_sha(inspect.getfile(Dinov2Model)),
       'strict_loading_info':info,'within_version_final_norm_maxabs':within_norm,'output_sha256':file_sha(target),
       'shapes':{k:list(v.shape) for k,v in arrays.items()},'finite':True,'CLS_drop_once':True,'final_patch_norm_eps':1e-8}
    save_json(WORK/'reports'/f'encoder-{args.label}.json',metadata);print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
