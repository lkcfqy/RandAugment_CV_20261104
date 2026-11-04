"""Static source and safe checkpoint metadata audit. Does not instantiate models."""
from counting_assets import ROOT, dump
import ast, math, struct, json, hashlib, collections, importlib.metadata as meta, sys, difflib
import safetensors
hroot=ROOT/'sources/countgdpp-hf';groot=ROOT/'sources/countgdpp-github'
expected={}
def tensor(k, shape, dtype='F32'):
    assert k not in expected
    expected[k]={'shape':shape,'dtype':dtype}
def linear(k,out,inp,bias=True):
    tensor(k+'.weight',[out,inp])
    if bias:tensor(k+'.bias',[out])
def norm(k,n):tensor(k+'.weight',[n]);tensor(k+'.bias',[n])
def mha(k,d):
    tensor(k+'.in_proj_weight',[3*d,d]);tensor(k+'.in_proj_bias',[3*d]);linear(k+'.out_proj',d,d)
def msattn(k):
    for n,o in [('sampling_offsets',256),('attention_weights',128),('value_proj',256),('output_proj',256)]:linear(k+'.'+n,o,256)
def mlp(k,dims):
    for i,(inp,out) in enumerate(zip(dims,dims[1:])):linear(k+'.layers.'+str(i),out,inp)

# Swin-B-384: source depths 2,2,18,2, widths 128,256,512,1024; window 12.
tensor('backbone.0.patch_embed.proj.weight',[128,3,4,4]);tensor('backbone.0.patch_embed.proj.bias',[128]);norm('backbone.0.patch_embed.norm',128)
for stage,depth in enumerate([2,2,18,2]):
    c=128*2**stage;heads=[4,8,16,32][stage]
    for block in range(depth):
        k=f'backbone.0.layers.{stage}.blocks.{block}'
        norm(k+'.norm1',c);norm(k+'.norm2',c)
        tensor(k+'.attn.relative_position_bias_table',[529,heads]);tensor(k+'.attn.relative_position_index',[144,144],'I64')
        linear(k+'.attn.qkv',3*c,c);linear(k+'.attn.proj',c,c);linear(k+'.mlp.fc1',4*c,c);linear(k+'.mlp.fc2',c,4*c)
    if stage<3:norm(f'backbone.0.layers.{stage}.downsample.norm',4*c);linear(f'backbone.0.layers.{stage}.downsample.reduction',2*c,4*c,False)
    if stage in [1,2,3]:norm(f'backbone.0.norm{stage}',c)
# BERT-base complete embeddings + all 12 layers + pooler.
tensor('bert.embeddings.word_embeddings.weight',[30522,768]);tensor('bert.embeddings.position_embeddings.weight',[512,768]);tensor('bert.embeddings.token_type_embeddings.weight',[2,768]);norm('bert.embeddings.LayerNorm',768)
for i in range(12):
    k=f'bert.encoder.layer.{i}'
    for n in ['query','key','value']:linear(k+'.attention.self.'+n,768,768)
    linear(k+'.attention.output.dense',768,768);norm(k+'.attention.output.LayerNorm',768)
    linear(k+'.intermediate.dense',3072,768);linear(k+'.output.dense',768,3072);norm(k+'.output.LayerNorm',768)
linear('bert.pooler.dense',768,768)
# 6 native encoder/decoder layers with text enhancer and bidirectional fusion.
for i in range(6):
    k=f'transformer.encoder.layers.{i}';msattn(k+'.self_attn');linear(k+'.linear1',2048,256);linear(k+'.linear2',256,2048);norm(k+'.norm1',256);norm(k+'.norm2',256)
    k=f'transformer.encoder.text_layers.{i}';mha(k+'.self_attn',256);linear(k+'.linear1',1024,256);linear(k+'.linear2',256,1024);norm(k+'.norm1',256);norm(k+'.norm2',256)
    k=f'transformer.encoder.fusion_layers.{i}'
    norm(k+'.layer_norm_v',256);norm(k+'.layer_norm_l',256);tensor(k+'.gamma_v',[256]);tensor(k+'.gamma_l',[256])
    for n in ['v_proj','l_proj','values_v_proj','values_l_proj']:linear(k+'.attn.'+n,1024,256)
    for n in ['out_v_proj','out_l_proj']:linear(k+'.attn.'+n,256,1024)
    k=f'transformer.decoder.layers.{i}';mha(k+'.self_attn',256);mha(k+'.ca_text',256);msattn(k+'.cross_attn')
    for n in ['norm1','norm2','norm3','catext_norm']:norm(k+'.'+n,256)
    linear(k+'.linear1',2048,256);linear(k+'.linear2',256,2048)
norm('transformer.decoder.norm',256);mlp('transformer.decoder.ref_point_head',[512,256,256]);mlp('transformer.enc_out_bbox_embed',[256,256,256,4]);linear('transformer.enc_output',256,256);norm('transformer.enc_output_norm',256);tensor('transformer.level_embed',[4,256]);tensor('transformer.tgt_embed.weight',[900,256])
tensor('feature_map_proj.weight',[256,1792,1,1]);tensor('feature_map_proj.bias',[256]);linear('feat_map',256,768)
for i,c in enumerate([256,512,1024,1024]):
    tensor(f'input_proj.{i}.0.weight',[256,c,3 if i==3 else 1,3 if i==3 else 1]);tensor(f'input_proj.{i}.0.bias',[256]);norm(f'input_proj.{i}.1',256)
mlp('bbox_embed.0',[256,256,256,4])
aliases={}
for root in ['bbox_embed','transformer.decoder.bbox_embed']:
    for i in range(6):
        if root=='bbox_embed' and i==0:continue
        for layer in range(3):
            for wb in ['weight','bias']:aliases[f'{root}.{i}.layers.{layer}.{wb}']=f'bbox_embed.0.layers.{layer}.{wb}'

p=ROOT/'models/model.safetensors'
with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
actual={k:{'dtype':v['dtype'],'shape':v['shape']} for k,v in header.items() if k!='__metadata__'}
checks={'missing_physical_keys':sorted(set(expected)-set(actual)),'unexpected_physical_keys':sorted(set(actual)-set(expected)),'shape_dtype_mismatches':[k for k in expected.keys()&actual.keys() if expected[k]!=actual[k]],'alias_metadata_matches_native_shared_bbox_head':header.get('__metadata__')==aliases}
with safetensors.safe_open(str(p),framework='numpy') as f:
    assert set(f.keys())==set(actual)
    for k,v in actual.items():assert f.get_slice(k).get_shape()==v['shape']
dump('reports/state_manifest.json',{'model_file':'models/model.safetensors','tensor_metadata':header,'metadata_only':True,'source_code_executed':False,'expected_schema':expected,'expected_shared_aliases':aliases,'static_comparison':checks})
summary={'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'file_bytes':p.stat().st_size,'header_bytes':n,'physical_tensor_count':len(actual),'shared_alias_count':len(aliases),'expanded_state_key_count':len(actual)+len(aliases),'unique_parameter_elements':sum(math.prod(v['shape']) for v in actual.values() if v['dtype']=='F32'),'integer_buffer_elements':sum(math.prod(v['shape']) for v in actual.values() if v['dtype']=='I64'),'tensor_metadata_read_with':'safetensors.safe_open framework=numpy; get_slice/get_shape only; no full tensors or model','static_comparison':checks,'backbone_complete_by_static_schema':all(actual[k]==v for k,v in expected.items() if k.startswith('backbone.')),'bert_complete_by_static_schema':all(actual[k]==v for k,v in expected.items() if k.startswith('bert.')),'strict_runtime_load_executed':False,'runtime_equivalence_proven':False,'forward_calls':0,'gpu_operations':0,'safetensors_version':safetensors.__version__}
dump('reports/model_static_verification.json',summary)

# Static Python import closure, including package __init__ imports. No exec/import of repo code.
mods={str(p.relative_to(hroot))[:-3].replace('/','.').removesuffix('.__init__'):p for p in hroot.rglob('*.py')}
local_roots={x.split('.')[0] for x in mods}; visited=set();edges=[];external=collections.defaultdict(set);deferred=collections.defaultdict(set)
def imports(nodes,defer=False):
    for node in nodes:
        if isinstance(node,(ast.Import,ast.ImportFrom)):yield node,defer
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):yield from imports(node.body,True)
        elif isinstance(node,ast.ClassDef):yield from imports(node.body,defer)
        elif isinstance(node,ast.If):
            # Importing a module never executes its __main__ debugging block.
            if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__' and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Constant) and node.test.comparators[0].value=='__main__':
                yield from imports(node.body,True);yield from imports(node.orelse,defer)
            else:yield from imports(node.body,defer);yield from imports(node.orelse,defer)
        elif isinstance(node,ast.Try):
            yield from imports(node.body,defer);yield from imports(node.orelse,defer);yield from imports(node.finalbody,defer)
            for h in node.handlers:yield from imports(h.body,defer)
def recurse(mod):
    if mod in visited:return
    visited.add(mod);path=mods[mod]
    parts=mod.split('.')
    for i in range(1,len(parts)):
        parent='.'.join(parts[:i])
        if parent in mods:recurse(parent)
    package=parts if path.name=='__init__.py' else parts[:-1]
    for node,is_deferred in imports(ast.parse(path.read_text()).body):
        candidates=[]
        if isinstance(node,ast.Import):candidates=[n.name for n in node.names]
        else:
            base='.'.join(package[:len(package)-node.level+1]+([node.module] if node.module else [])) if node.level else node.module or ''
            candidates=[base]+[base+'.'+n.name for n in node.names]
        for name in candidates:
            if not name:continue
            if name in mods:
                edges.append({'from':mod,'to':name,'line':node.lineno,'deferred':is_deferred})
                if not is_deferred:recurse(name)
            elif name.split('.')[0] not in local_roots:
                root=name.split('.')[0]
                (deferred if is_deferred else external)[root].add(mod)
recurse('countgdpp')
stdlib=set(sys.stdlib_module_names)
ext={k:sorted(v) for k,v in external.items() if k not in stdlib}
dump('reports/import_closure.json',{'entrypoint':'countgdpp.py','method':'AST only, eager imports plus package __init__ closure; deferred imports listed separately','modules':sorted(visited),'edges':edges,'external_import_roots':ext,'deferred_external_import_roots':{k:sorted(v) for k,v in deferred.items() if k not in stdlib},'repository_modules_executed':0,'optional_cuda_extension':'MultiScaleDeformableAttention import under try/except; _C=None on absence.'})

comparison={'same':[],'different':[],'hf_only':[]}
patch=[]
for f in sorted(hroot.rglob('*.py')):
    rel=f.relative_to(hroot);g=groot/rel
    if not g.exists():comparison['hf_only'].append(str(rel))
    elif g.read_bytes()==f.read_bytes():comparison['same'].append(str(rel))
    else:
        comparison['different'].append(str(rel));patch.extend(difflib.unified_diff(g.read_text().splitlines(True),f.read_text().splitlines(True),fromfile='github/'+str(rel),tofile='hf/'+str(rel)))
dump('reports/source_identity_comparison.json',comparison)
(ROOT/'reports/github_to_hf.diff').write_text(''.join(patch))
versions={}
for name in ['torch','torchvision','transformers','huggingface_hub','safetensors','timm','addict','yapf','termcolor','opencv-python','opencv-python-headless','pycocotools','scipy','numpy','pillow','matplotlib','pyyaml','tokenizers']:
    try:versions[name]=meta.version(name)
    except meta.PackageNotFoundError:versions[name]=None
dump('reports/existing_environment.json',{'python':sys.version,'executable':sys.executable,'installed_versions':versions,'new_environment_created':False,'packages_installed':0,'known_conflict':'Installed transformers 4.57.1 violates official HF transformers<4.50. No old environment changed.','missing_packages':[x for x in ['timm','addict','yapf','termcolor','opencv-python-headless','pycocotools'] if versions[x] is None]})
# Preserve installed dependency source used to inspect strict/tied loading and ROIAlign dispatch.
for package,relative,name in [('huggingface_hub','huggingface_hub/hub_mixin.py','installed-hf-hub-mixin.py.txt'),('safetensors','safetensors/torch.py','installed-safetensors-torch.py.txt'),('torchvision','torchvision/ops/roi_align.py','installed-torchvision-roi_align.py.txt')]:
    src=meta.distribution(package).locate_file(relative);(ROOT/'sources'/name).write_bytes(src.read_bytes())
print(json.dumps({'model':summary,'eager_modules':len(visited),'external_import_roots':sorted(ext),'same_source_files':len(comparison['same'])},ensure_ascii=False))
