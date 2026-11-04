"""Pinned existing official CPU runtime and explicit original initialization contract."""
from contracts import *
import sys,random,copy,importlib.util
F=R/'fixed-M-execution-20261101/payload'
sys.path.insert(0,str(F))
from candidate_control import validate_optimizer_groups

def normalize_config(value):
    if isinstance(value,list):return [normalize_config(x) for x in value]
    if not isinstance(value,dict):return value
    result={k:normalize_config(v) for k,v in value.items()}
    for key in ('classes','scale','crop_size'):
        if isinstance(result.get(key),list):result[key]=tuple(result[key])
    if 'scales' in result:result['scales']=[tuple(x) if isinstance(x,list) else x for x in result['scales']]
    return result

def setup():
    import torch,numpy as np,cv2,mmcv,mmdet,mmengine,torchvision
    from mmdet.utils import register_all_modules
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
    assert torch.version.cuda is None and torch.get_num_threads()==4 and torch.get_num_interop_threads()==1 and cv2.getNumThreads()==1
    register_all_modules(init_default_scope=True);mmcv.use_backend('cv2')
    proto=read(P/'PROTOCOL.json')
    for n,h in proto['external_local_code_pins'].items():assert sha(R/n)==h,n
    actual={'torch':torch.__version__.split('+')[0],'torchvision':torchvision.__version__.split('+')[0],'numpy':np.__version__,'cv2':cv2.__version__,'mmdet':mmdet.__version__,'mmcv':mmcv.__version__,'mmengine':mmengine.__version__}
    assert actual==proto['core_runtime_versions']
    site=Path(mmdet.__file__).parent.parent
    for n,h in proto['official_implementation_sha256'].items():assert sha(site/n)==h,n
    assert sha(R/'cpu-cross-input-preflight-20261101/ENVIRONMENT_LOCK.json')==proto['environment_lock_sha256']
    return actual

def reset_seed():
    import torch,numpy as np
    random.seed(SEED);np.random.seed(SEED);torch.random.default_generator.manual_seed(SEED)

def state_hash(state):
    h=hashlib.sha256();size=0
    for n,t in sorted(state.items()):
        b=t.detach().contiguous().numpy().tobytes()
        h.update(json.dumps([n,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(b);size+=len(b)
    return {'state_sha256':h.hexdigest(),'tensor_count':len(state),'tensor_bytes':size}

def build_model(checkpoint=None):
    import torch
    from mmengine.config import Config
    from mmdet.registry import MODELS
    proto=read(P/'PROTOCOL.json');ref=read(R/'unadapted-reference-20261101/PROTOCOL.json')
    for a in proto['asset_pins']:
        f=R/a['path'];assert f.stat().st_size==a['bytes'] and sha(f)==a['sha256']
    for a in read(R/'unadapted-reference-20261101/INITIALIZATION_SEMANTICS.json')['static_sources']:
        assert sha(a['path'])==a['sha256']
    conf=F/'config_A_reference.json';assert sha(conf)==proto['historical_A_config_sha256']
    cfg=Config(normalize_config(read(conf)));cfg['model']['language_model']['name']=str(R/'assets/bert-base-uncased')
    assert cfg['param_scheduler']==[] and cfg['model']['bbox_head']['num_classes']==1
    t=time.monotonic();model=MODELS.build(cfg['model']);model.init_weights();model.to(torch.device('cpu'))
    source_path=R/proto['asset_pins'][0]['path'] if checkpoint is None else Path(checkpoint)
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    assert isinstance(source,dict) and all(isinstance(v,torch.Tensor) for v in source.values())
    assert all(bool(torch.isfinite(v).all()) for v in source.values() if v.is_floating_point())
    state=model.state_dict()
    if checkpoint is None:
        common=set(state)&set(source);missing=set(state)-set(source);extra=set(source)-set(state)
        logs=set(ref['missing_model_definitions']['fixed_identity_log_scales']);dn=ref['missing_model_definitions']['random_training_only_DN']
        assert len(source)==903 and len(state)==908 and len(common)==900
        assert missing==logs|{dn} and extra==set(ref['extra_source_keys'])
        assert all(tuple(state[n].shape)==(1,) and state[n].dtype==torch.float32 and state[n].item()==0 for n in logs)
        assert tuple(state[dn].shape)==(1,256)
        assert model.language_model.language_backbone.body.model.pooler is None
        assert all(b.bias is None for b in model.bbox_head.cls_branches)
        assert torch.equal(source['language_model.language_backbone.body.model.embeddings.position_ids'],torch.arange(512).reshape(1,512))
        complete={n:source[n] if n in common else state[n] for n in state}
    else:
        assert set(source)==set(state) and len(source)==908
        common=set(state);complete=source
    for n in common:assert source[n].shape==state[n].shape and source[n].dtype==state[n].dtype
    inc=model.load_state_dict(complete,strict=True);assert not inc.missing_keys and not inc.unexpected_keys
    loaded=model.state_dict()
    for n in common:assert torch.equal(loaded[n],source[n])
    initial=state_hash(loaded)
    if checkpoint is None:assert initial==ref['expected_complete_initial_state']
    assert all(t.device.type=='cpu' for t in list(model.parameters())+list(model.buffers()))
    report={'seconds':time.monotonic()-t,'source_sha256':sha(source_path),'load_state':initial,'weights_only':True,'strict_complete_state':True,'raw_original_direct_strict':False if checkpoint is None else None,'raw_source_keys':len(source),'no_unsafe_fallback':True}
    del source,state,loaded,complete
    return model,cfg,report

def observer():
    path=R/'ets-pipeline-dryrun-20261101/dryrun.py'
    assert sha(path)==read(P/'PROTOCOL.json')['observer_source_sha256']
    spec=importlib.util.spec_from_file_location('brackish_frozen_observer',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.HERE=P
    assert read(F/'config_A_reference.json')['train_dataloader']['dataset']['pipeline'][0]['transforms']==mod.PROTO['arms']['A']
    return mod

def dataset(group):
    from mmdet.datasets import CocoDataset
    return CocoDataset(ann_file=str(P/f'data/annotations/{group}.json'),data_root=str(P/'data')+'/',data_prefix={'img':'images/'},metainfo={'classes':('fish',)},filter_cfg={'filter_empty_gt':False,'min_size':0},return_classes=True,pipeline=[],test_mode=group in ('dev','confirm'))

def training_order(ds):
    from mmengine.dataset import DefaultSampler
    s=DefaultSampler(ds,shuffle=True,seed=SEED,round_up=False);order=[]
    for epoch in range(40):s.set_epoch(epoch);order.extend(list(s))
    assert len(order)==200 and all(order.count(i)==40 for i in range(5))
    return order

def check_code_pins():
    frozen=read(P/'CODE_PINS.json')
    for name,h in frozen.items():assert sha(P/name)==h,name
