"""Strict fixed safetensors loading, adapted from the already audited CPU loader."""
import hashlib,json,sys
from pathlib import Path

def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load_fixed_model(root):
    import torch
    from safetensors.torch import load_file
    root=Path(root);source=root/'source'
    identities=json.loads((root/'manifests/model_identity.json').read_text())
    for rel,sha in identities['source_files'].items():
        if file_hash(source/rel)!=sha:raise ValueError('official source hash mismatch: '+rel)
    for item in identities['weight_assets']:
        if file_hash(root/item['path'])!=item['sha256']:raise ValueError('safe weight hash mismatch')
    sys.path.insert(0,str(source))
    import importlib.util
    if importlib.util.find_spec('MultiScaleDeformableAttention') is not None:
        raise RuntimeError('unexpected custom attention extension; require natural official PyTorch path')
    from countgdpp import CountGDPlusPlus,count
    from models.GroundingDINO import ms_deform_attn
    if ms_deform_attn._C is not None:raise RuntimeError('custom CUDA extension unexpectedly loaded')
    config=json.loads((source/'config.json').read_text());config['device']='cpu';config['text_encoder_type']=str(root/'models/bert-base-uncased')
    if 'auto_map' in json.loads((root/'models/bert-base-uncased/config.json').read_text()):raise RuntimeError('remote model code prohibited')
    model=CountGDPlusPlus(config=config).eval().requires_grad_(False)
    manifest=json.loads((root/'manifests/model_schema.json').read_text());aliases=manifest['expected_shared_aliases']
    physical=load_file(str(root/'models/model.safetensors'),device='cpu');state=model.state_dict()
    if set(physical)!=set(manifest['expected_schema']) or set(state)!=set(physical)|set(aliases):raise ValueError('checkpoint logical/physical key mismatch')
    registered=dict(model.named_parameters(remove_duplicate=False));registered.update(dict(model.named_buffers(remove_duplicate=False)))
    nonpersistent={name+'.'+key for name,module in model.named_modules() for key in module._non_persistent_buffers_set if module._buffers.get(key) is not None}
    if set(registered)-set(state)!=nonpersistent or nonpersistent!={'bert.embeddings.position_ids','bert.embeddings.token_type_ids'} or set(state)-set(registered):raise ValueError('unaccounted registered tensor')
    if not torch.equal(registered['bert.embeddings.position_ids'],torch.arange(512).expand(1,-1)) or not torch.equal(registered['bert.embeddings.token_type_ids'],torch.zeros((1,512),dtype=torch.long)):raise ValueError('nonpersistent BERT cache mismatch')
    def storage(t):return t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),t.dtype,t.device
    for alias,canonical in aliases.items():
        if alias in physical or canonical not in physical or registered[alias] is not registered[canonical] or storage(state[alias])!=storage(state[canonical]) or manifest['tensor_metadata']['__metadata__'][alias]!=canonical:raise ValueError('actual alias mismatch: '+alias)
    payload=dict(physical);payload.update({alias:physical[canonical] for alias,canonical in aliases.items()})
    for key,value in payload.items():
        if value.shape!=state[key].shape or value.dtype!=state[key].dtype or value.device.type!='cpu' or state[key].device.type!='cpu':raise ValueError('shape/dtype/device mismatch: '+key)
    if not all(bool(torch.isfinite(value).all()) for value in physical.values()):raise ValueError('nonfinite checkpoint')
    loaded=model.load_state_dict(payload,strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:raise ValueError('strict load unexpectedly incomplete')
    if not all(torch.equal(value,physical[aliases.get(key,key)]) for key,value in model.state_dict().items()):raise ValueError('loaded tensor differs from checkpoint')
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('model not fixed/eval')
    report={'logical_keys':len(state),'physical_keys':len(physical),'actual_aliases':len(aliases),'missing_keys':[],'unexpected_keys':[],'strict':True,'all_values_equal':True,'custom_attention_extension':False}
    return model,count,aliases,report

def registered_snapshot(model,aliases):
    from hooks import tensor_hash
    tensors=dict(model.named_parameters(remove_duplicate=False));tensors.update(dict(model.named_buffers(remove_duplicate=False)))
    device=next(model.parameters()).device
    if any(value.device!=device for value in tensors.values()):raise ValueError('registered tensors are on mixed devices')
    for alias,canonical in aliases.items():
        if tensors[alias] is not tensors[canonical]:raise ValueError('shared Parameter identity lost in migration')
        a,b=tensors[alias],tensors[canonical]
        if (a.untyped_storage().data_ptr(),a.storage_offset(),a.shape,a.stride(),a.dtype,a.device)!=(b.untyped_storage().data_ptr(),b.storage_offset(),b.shape,b.stride(),b.dtype,b.device):raise ValueError('shared storage lost in migration')
    return {'tensor_sha256':{k:tensor_hash(v) for k,v in tensors.items()},'actual_aliases':aliases,'registered_devices_consistent':True}
