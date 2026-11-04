"""Compare the selected built-in eager path with the official 4.46.3 formulas.

Downloaded source is parsed, never imported or executed. Synthetic inputs only.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ast,inspect,json,math,textwrap,importlib.metadata
import torch
import torch.nn.functional as F
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
from stheta.encoder import load_encoder
from stheta.protocol import ASSETS,WORK,REVISION,WEIGHT_SHA,save_json,file_sha

def method_body(source,classname,name):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    body=fn.body
    if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):body=body[1:]
    return ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)

@torch.inference_mode()
def reference_446_eager(model,x):
    # Explicit no-mask eval path, corresponding to official v4.46.3.
    B,C,H,W=x.shape;e=model.embeddings
    z=e.patch_embeddings.projection(x).flatten(2).transpose(1,2)
    z=torch.cat((e.cls_token.expand(B,-1,-1),z),dim=1)
    pos=e.position_embeddings;D=z.shape[-1];side=int((pos.shape[1]-1)**.5)
    p=pos[:,1:].reshape(1,side,side,D).permute(0,3,1,2)
    p=F.interpolate(p.float(),size=(H//14,W//14),mode='bicubic',align_corners=False).to(pos.dtype)
    p=torch.cat((pos[:,:1],p.permute(0,2,3,1).reshape(1,-1,D)),dim=1)
    z=z+p
    for layer in model.encoder.layer:
        n=layer.norm1(z);a=layer.attention.attention;heads=a.num_attention_heads;hd=a.attention_head_size
        q=a.query(n).view(B,-1,heads,hd).permute(0,2,1,3)
        k=a.key(n).view(B,-1,heads,hd).permute(0,2,1,3)
        v=a.value(n).view(B,-1,heads,hd).permute(0,2,1,3)
        probs=torch.softmax((q@k.transpose(-1,-2))/math.sqrt(hd),dim=-1)
        context=(probs@v).permute(0,2,1,3).contiguous().view(B,-1,D)
        attn=layer.attention.output.dense(context)
        z=z+attn*layer.layer_scale1.lambda1
        n=layer.norm2(z);mlp=layer.mlp.fc2(F.gelu(layer.mlp.fc1(n),approximate='none'))
        z=z+mlp*layer.layer_scale2.lambda1
    return model.layernorm(z),p

def main():
    torch.set_num_threads(4);torch.manual_seed(441)
    old=(ASSETS/'sources/transformers-dinov2-4.46.3.py.txt').read_text()
    new=(ASSETS/'sources/transformers-dinov2-4.57.1.py.txt').read_text()
    body_old=method_body(old,'Dinov2Embeddings','interpolate_pos_encoding')
    body_new=method_body(new,'Dinov2Embeddings','interpolate_pos_encoding')
    installed='class Dinov2Embeddings:\n'+textwrap.indent(textwrap.dedent(inspect.getsource(Dinov2Embeddings.interpolate_pos_encoding)),'    ')
    body_installed=method_body(installed,'Dinov2Embeddings','interpolate_pos_encoding')
    assert body_old==body_new==body_installed
    assert json.loads((ASSETS/'sources/hf-protocol-config.json').read_text())==json.loads((ASSETS/'models/dinov2-small/config.json').read_text())
    assert 'oid sha256:'+WEIGHT_SHA in (ASSETS/'sources/hf-protocol-safetensors-lfs.txt').read_text()
    model=load_encoder();x=torch.stack((torch.zeros(3,224,224),torch.randn(3,224,224)))
    with torch.inference_mode():
        actual=model(pixel_values=x,output_hidden_states=True,return_dict=True)
        expected,pos=reference_446_eager(model,x)
        diff=float((actual.last_hidden_state-expected).abs().max())
        nativepos=model.embeddings.interpolate_pos_encoding(torch.empty(2,257,384),224,224)
        posdiff=float((nativepos-pos).abs().max())
        normdiff=float((actual.last_hidden_state-model.layernorm(actual.hidden_states[-1])).abs().max())
        assert torch.allclose(actual.last_hidden_state,expected,rtol=1e-5,atol=3e-5)
        assert posdiff==0 and normdiff==0
    report={'status':'passed','installed_transformers':importlib.metadata.version('transformers'),
      'comparison_version':'4.46.3','attention':'eager only, FP32 eval, no mask/head mask/dropout',
      'pos_interpolation_AST_identical':True,'native_pos_max_abs':posdiff,'final_norm_max_abs':normdiff,
      'full_legacy_formula_max_abs':diff,'synthetic_cases':['zeros','seed441_normal'],
      'equivalence_scope':'Selected Dinov2-small FP32 eager eval path has matching formulas and passes synthetic numerical comparison; no claim for other modes, dtype, backends or general bitwise identity',
      'new_attention_refactor':'v4.57.1 uses shared eager function; scale=64**(-.5), softmax float32, same no-dropout forward mathematics',
      'weights_original_asset_revision':'ed25f3a31f01632728cabb09d1542f84ab7b0056','protocol_revision':REVISION,
      'same_weight_sha256':WEIGHT_SHA,'same_config':True,'weight_redownloaded':False,'environment_downgraded':False,
      'model_eager_config':model.config._attn_implementation,'sources':{v:file_sha(ASSETS/f'sources/transformers-dinov2-{v}.py.txt') for v in ['4.46.3','4.57.1']}}
    save_json(WORK/'reports/transformers-equivalence.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
