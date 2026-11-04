"""Audited pinned CIDNet adapter. Import is inert; construction requires release."""
import json
import os
import sys
from pathlib import Path
import numpy as np
from .core import rgb,finite
from .audit import require_release,file_sha

SOURCE_COMMIT='eb43d7d91e9a336c66856824ff9e4603ae41f408'
SOURCE_SHA256={
 'net/CIDNet.py':'dc9b9232f225f5760417175d25f8efc146f51c0acd1ee014df6b9f0a6b2e8d84',
 'net/HVI_transform.py':'8be0fea6e61c51d6290722f0cf81be1f334b1b998dbfd409435343d9d4bd5bb2',
 'net/transformer_utils.py':'7bfb2659e97289f0e8005935e15edbe0af3ae5ef05c50863a074fd745617b13f',
 'net/LCA.py':'271f487091bd13ed211a8b0c9cb07036957ba535f5dd3aedc131498500979d26',
}
WEIGHT_SHA='2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894'
CONFIG={'channels':[36,36,72,144],'heads':[1,2,4,8],'norm':False}

class CIDNetPredictor:
    def __init__(self,asset_root,*,device,receipt,freeze_sha):
        require_release(receipt,freeze_sha)
        if receipt.get('scope') not in ('preflight','formal'):raise ValueError('No model execution scope')
        if device not in ('cpu','mps','cuda'):raise ValueError('Explicit supported device required')
        root=Path(asset_root).resolve();src=root/'vendor/HVI-CIDNet'
        for rel,expected in SOURCE_SHA256.items():
            if file_sha(src/rel)!=expected:raise ValueError('Code differs from fixed official commit: '+rel)
        if file_sha(root/'models/model.safetensors')!=WEIGHT_SHA:raise ValueError('Model checksum mismatch')
        if json.loads((root/'models/config.json').read_text())!=CONFIG:raise ValueError('Model config mismatch')
        for name,module in tuple(sys.modules.items()):
            if name.startswith('net.') and getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(src):raise ValueError('Another net package is already loaded')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        import torch
        from safetensors.torch import load_file
        sys.path.insert(0,str(src))
        from net.CIDNet import CIDNet
        for rel in SOURCE_SHA256:
            mod=sys.modules[rel[:-3].replace('/','.')]
            if Path(mod.__file__).resolve()!=(src/rel).resolve():raise ValueError('Imported code path mismatch')
        state=load_file(str(root/'models/model.safetensors'),device='cpu')
        if len(state)!=191 or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in state.values()):raise ValueError('Invalid 191-key finite FP32 checkpoint')
        with torch.device('cpu'):self.model=CIDNet(**CONFIG)
        result=self.model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys:raise ValueError('Strict state mismatch')
        self.model.trans.gated=True;self.model.trans.gated2=True
        self.model.trans.alpha_s=1.0;self.model.trans.alpha=1.0
        self.gamma=1.0;self.device=device;self.torch=torch
        self.model.eval().requires_grad_(False).to(device=device,dtype=torch.float32)
        self.calls=0
    def __call__(self,image):
        image=rgb(image,quantized=True)
        if image.shape!=(3,400,600):raise ValueError('Native 600x400 batch-one input required; no resize/crop/pad')
        tensor=self.torch.from_numpy(np.array(image,dtype=np.float32,order='C',copy=True)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            out=self.model(tensor**self.gamma)
            if out.shape!=(1,3,400,600) or not self.torch.isfinite(out).all():raise ValueError('Invalid model output')
            out=out.clamp(0,1)
        self.calls+=1
        return out[0].detach().cpu().numpy()
    def synchronize(self):
        if self.device=='mps':self.torch.mps.synchronize()
        elif self.device=='cuda':self.torch.cuda.synchronize()
