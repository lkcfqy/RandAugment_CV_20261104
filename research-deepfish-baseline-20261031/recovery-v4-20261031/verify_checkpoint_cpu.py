"""Inspect fixed source and tensor state only; never imports or instantiates a detector."""
from portable_dev.paths import project_path as _portable_project_path
import ast, datetime, hashlib, json, pathlib
import torch
ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = _portable_project_path(__file__, 'research-deepfish-baseline-20261031')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()
sources=json.loads((ROOT/'sources/OFFICIAL_COMPATIBILITY_SOURCES.json').read_text())
for row in sources:
    p=ROOT/'sources'/(row['name']+'.py')
    assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    ast.parse(p.read_text())
def source(name): return (ROOT/'sources'/(name+'.py')).read_text()
cfg_tree=ast.parse(source('official_config'))
def literal(node):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
        return {k.arg:literal(k.value) for k in node.keywords}
    if isinstance(node,ast.Name) and node.id=='lang_model_name':return 'bert-base-uncased'
    if isinstance(node,ast.List):return [literal(x) for x in node.elts]
    if isinstance(node,ast.Tuple):return tuple(literal(x) for x in node.elts)
    return ast.literal_eval(node)
assign={n.targets[0].id:n.value for n in cfg_tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
official_model=literal(assign['model'])
cfg=json.loads((ROOT/'payload/config_A.json').read_text())['model']
for k in ['decoder','with_box_refine','as_two_stage']:
    assert cfg[k]==official_model[k]
assert cfg['bbox_head']['contrastive_cfg']==official_model['bbox_head']['contrastive_cfg']==dict(max_text_len=256,log_scale=0.0,bias=False)
assert cfg['language_model']['add_pooling_layer'] is official_model['language_model']['add_pooling_layer'] is False
assert 'groundingdino_swint_ogc_mmdet-822d7e9d.pth' in literal(assign['load_from'])
head=source('official_head');det=source('official_deformable_detr');bert=source('official_bert');hf=source('official_transformers_bert')
assert "bbox_head['share_pred_layer'] = not with_box_refine" in det
assert "bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1)" in det
assert 'if isinstance(log_scale, float):' in head and 'torch.Tensor([float(log_scale)]), requires_grad=True)' in head
assert 'res = res * self.log_scale.exp()' in head and 'copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)' in head
assert 'name, add_pooling_layer=add_pooling_layer, config=config)' in bert and 'outputs.hidden_states[1:]' in bert
assert 'self.pooler = BertPooler(config) if add_pooling_layer else None' in hf
assert 'pooled_output = self.pooler(sequence_output) if self.pooler is not None else None' in hf
hf_tree=ast.parse(hf)
position_calls=[n for n in ast.walk(hf_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='register_buffer' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='position_ids']
assert len(position_calls)==1 and any(k.arg=='persistent' and ast.literal_eval(k.value) is False for k in position_calls[0].keywords)
loader=ast.parse(source('official_mmengine_loader'))
strict_defaults={}
for n in loader.body:
    if isinstance(n,ast.FunctionDef) and n.name in ('load_checkpoint','_load_checkpoint_to_model','load_state_dict'):
        defaults=dict(zip([a.arg for a in n.args.args][-len(n.args.defaults):],n.args.defaults))
        strict_defaults[n.name]=ast.literal_eval(defaults['strict'])
assert strict_defaults==dict(load_checkpoint=False,_load_checkpoint_to_model=False,load_state_dict=False)
checkpoint=PROJECT/'assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
assert checkpoint.stat().st_size==691901857 and sha(checkpoint)=='822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
assert len(state)==903 and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in state.values())
assert all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point())
logkeys=['bbox_head.cls_branches.'+str(i)+'.log_scale' for i in range(7)]
assert not set(logkeys)&set(state) and 'dn_query_generator.label_embedding.weight' not in state
prefix='language_model.language_backbone.body.model.'
assert torch.equal(state[prefix+'embeddings.position_ids'],torch.arange(512).reshape(1,512))
for suffix,shape in [('weight',(768,768)),('bias',(768,))]:assert tuple(state[prefix+'pooler.dense.'+suffix].shape)==shape
original=PROJECT/'local-runpack-20261031/payload'
unchanged=['config_A.json','config_B.json','asset_pins.json','implementation_pins.json','data_selection.json','data/annotations/support5.json','data/annotations/dev64.json']
for name in unchanged: assert (ROOT/'payload'/name).read_bytes()==(original/name).read_bytes()
oldtree=ast.parse((original/'contracts.py').read_text());newtree=ast.parse((ROOT/'payload/contracts.py').read_text())
oldfuncs={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)}
newfuncs={n.name:ast.dump(n,include_attributes=False) for n in newtree.body if isinstance(n,ast.FunctionDef)}
scientific_functions=['choose_updates','verify_exposures','verify_paired_traces','verify_initializations']
for name in scientific_functions:assert oldfuncs[name]==newfuncs[name], name
report={'status':'PASS_CPU_STATIC_AND_SAFE_TENSOR_INSPECTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'torch':torch.__version__,'model_instantiated':False,'upstream_code_executed':False,'weights_only':True,'unsafe_fallback':False,'dynamic_whitelist':False,'checkpoint_sha256':sha(checkpoint),'source_tensor_keys':len(state),'all_source_floating_tensors_finite':True,'official_config_same_checkpoint':True,'official_log_scale_zero_initializers':logkeys,'log_scale_semantics':'7 independent trainable float32 scalar parameters, initialized 0.0; exp(0)=1.0; official 6 decoder layers plus encoder prediction layer.','unused_pooler_shapes':{'weight':[768,768],'bias':[768]},'pooler_semantics':'official config disables construction, MMDetection consumes encoder hidden states; no pooler output consumed','legacy_position_ids':'exact arange(512).reshape(1,512); Transformers4.34 registers corresponding buffer persistent=False','official_loader_strict_defaults':strict_defaults,'loading_contract':'Official non-strict loader plus independent exact expected 8-missing/3-extra schema; all 900 common tensors must match exactly before any update. Not a claim of strict=True loading.','unchanged_scientific_files':{n:sha(ROOT/'payload'/n) for n in unchanged},'scientific_contract_functions_ast_unchanged':scientific_functions,'infrastructure_context_validation_revised':True,'sources':sources,'next_runtime_gate':'Check 7 zeros, requires_grad, exact expected key sets and all 900 source/model tensors; compare full initial state digest across four training workers.'}
(ROOT/'reports/CPU_CHECKPOINT_COMPATIBILITY.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','checkpoint_sha256','source_tensor_keys','model_instantiated','loading_contract']}))
