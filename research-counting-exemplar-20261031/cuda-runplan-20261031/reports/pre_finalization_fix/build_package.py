"""Local assets/code packaging only. No model construction, service calls, or GPU."""
from portable_dev.paths import project_path as _portable_project_path
import ast,gzip,hashlib,io,json,shutil,tarfile,time
from pathlib import Path

ROOT=_portable_project_path(__file__, 'research-counting-exemplar-20261031')
OUT=ROOT/'cuda-runplan-20261031';PAYLOAD=OUT/'payload';STAGE=Path('/tmp/counting_cuda')

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)

def main():
    for path in STAGE.glob('*.py'):
        ast.parse(path.read_text())
        if path.name.startswith('test_'):copy(path,OUT/'tests'/path.name)
        elif path.name in ('build_package.py','prepare_dependencies.py','verify_package.py'):copy(path,OUT/path.name)
        else:copy(path,PAYLOAD/'scripts'/path.name)
    source_hashes=json.loads((ROOT/'cpu-readiness/reports/cpu_verification.json').read_text())['source_sha256_before']
    for relative,expected in source_hashes.items():
        source=ROOT/'sources/countgdpp-hf'/relative
        if sha(source)!=expected:raise RuntimeError('official source changed')
        copy(source,PAYLOAD/'source'/relative)
    for path in (ROOT/'models/bert-base-uncased').iterdir():
        if path.is_file() and path.name!='model.safetensors':copy(path,PAYLOAD/'models/bert-base-uncased'/path.name)
    copy(ROOT/'reports/state_manifest.json',PAYLOAD/'manifests/model_schema.json')
    weights=[{'path':'models/model.safetensors','bytes':937560192,'sha256':'94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5','url':'https://huggingface.co/nikigoli/CountGDPlusPlus/resolve/06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1/model.safetensors'},
             {'path':'models/bert-base-uncased/model.safetensors','bytes':440449768,'sha256':'68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3','url':'https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/model.safetensors'}]
    write(PAYLOAD/'manifests/model_identity.json',{'HF_model':'nikigoli/CountGDPlusPlus','HF_revision':'06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1','BERT_revision':'86b5e0934494bd15c9632b12f734a8a67f723594','source_files':source_hashes,'weight_assets':weights,'safe_only':True,'source_altered':False})
    for file,source in [('selection_frozen.json',ROOT/'reports/selection_frozen.json'),('labels_frozen.json',ROOT/'observation-entry/labels_frozen.json'),('queries_frozen.json',ROOT/'observation-entry/queries_frozen.json'),('protocol_v2_frozen.json',ROOT/'observation-entry/protocol_v2_frozen.json')]:copy(source,PAYLOAD/'science'/file)
    selection=json.loads((PAYLOAD/'science/selection_frozen.json').read_text())
    for image in selection['images']:copy(ROOT/'data/images'/image['filename'],PAYLOAD/'science/images'/image['filename'])
    for name in ('LICENSE','README.md','dataset/README.md'):copy(ROOT/'sources/pairtally'/name,PAYLOAD/'licenses/pairtally'/name)
    write(PAYLOAD/'manifests/science_files.json',{str(path.relative_to(PAYLOAD/'science')):sha(path) for path in sorted((PAYLOAD/'science').rglob('*')) if path.is_file()})
    import sys
    sys.path.insert(0,str(STAGE));from fixture import generate,SEED,WIDTH,HEIGHT,POSITIVE,NEGATIVE,TEXTS
    pixels=generate();raw=pixels.tobytes();fixture_dir=PAYLOAD/'fixture';fixture_dir.mkdir(exist_ok=True)
    (fixture_dir/'texture.rgb').write_bytes(raw)
    write(fixture_dir/'manifest.json',{'version':'textured-toy-v1','generator_sha256':sha(PAYLOAD/'scripts/fixture.py'),'pixel_sha256':hashlib.sha256(raw).hexdigest(),'seed':SEED,'width':WIDTH,'height':HEIGHT,'pixel_bytes':len(raw),'dtype':'uint8','channels':'RGB','positive_xyxy':POSITIVE,'negative_xyxy':NEGATIVE,'texts':TEXTS,'fixture_selection':'single deterministic generator fixed before model execution; no alternatives/tuning','model_rng_seed':42,'channel_standard_deviation':[float(pixels[:,:,i].std()) for i in range(3)]})
    previous=54.91578541700437
    old=json.loads((ROOT/'observation-entry/protocol_v2_frozen.json').read_text())
    current=dict(old);current.update(status='CODE_FROZEN_PENDING_EXPLICIT_ALLOCATION_GO',engineering_status='CUDA_PRECHECK_NOT_RUN',formal_execution_authorized=False,precheck_wall_seconds_charged=previous,metric_code_sha256=sha(PAYLOAD/'scripts/metrics.py'),parent_execution_clarification='2026-10-31: cumulative compute starts at 54.915785417 s; C,V boundaries inclusive >=0.30; truth bound to frozen labels; retain old protocol unchanged for audit',historical_protocol_sha256=sha(ROOT/'observation-entry/protocol_v2_frozen.json'))
    current['gate_pair_condition']=dict(old['gate_pair_condition'],C_operator='>=',V_operator='>=')
    current['protocol_version']=3;current['protocol_role']='ACTUAL_CUDA_EXECUTION_CONTRACT'
    current['supersedes_protocol_sha256']=sha(ROOT/'observation-entry/protocol_v2_frozen.json')
    current['authorization_policy']='Frozen before effects. This document grants no runtime permission; separate explicit GO must match FREEZE.json.'
    write(PAYLOAD/'execution_contract.json',current)
    plan={'status':'PENDING_EXPLICIT_GO_NO_ALLOCATION','previous_compute_wall_seconds':previous,'total_compute_wall_seconds':1800,'precheck_attempt_cap':7,'precheck_wall_seconds':180,'precheck_order':['CPU_F','CPU_C','CPU_V','CUDA_F_warmup','CUDA_F','CUDA_C','CUDA_V'],'formal_query_count':492,'eligible_pairs':41,'eligible_images':82,'NA_pairs':6,'all_pairs_reported':47,
          'allocation_count_cap':1,'accelerator':'T4','allocation_wall_seconds_cap':2700,'controller_release_reserve_seconds':90,'credit_cap_CU_strict':2,'credit_rounding_buffer_CU':.05,'no_new_OAuth':True,'no_model_retry_or_fixture_switch':True,'RSS_bytes_cap':8*1024**3,'CUDA_bytes_cap':8*1024**3,'dtype':'float32','AMP':False,'TF32':False,'native_attention_extension_must_be_None':True,
          'compute_gate_formula':'used_including_prior_CPU_MPS_and_all_runtime_preparation + 0_extra_reload_or_warmup + 2*492*max(CUDA_F,C,V_complete_query_seconds) + 60 <= 1800','io_and_finalization_reserve_seconds':60,'same_allocation_model_reuse':True,
          'download_bytes_before_allocation':1608332982,'bundle_transfer_budget_bytes':200*1024**2,'result_download_reserve_bytes':256*1024**2,'download_cap_bytes':int(3.5*1024**3),'owned_disk_cap_bytes':6*1024**3,
          'numerical_test':'elementwise abs(CUDA-CPU) <= 1e-4 + 1e-3*abs(CPU); active logits and boxes finite, fixed unused token padding exactly -inf; native device masks identical; no reorder rescue','execution_contract_sha256':sha(PAYLOAD/'execution_contract.json'),
          'model_forward_count_this_preparation':0,'historical_forward_attempts':6,'historical_real_queries':0,'historical_training_steps':0}
    write(PAYLOAD/'plan.json',plan)
    files={str(path.relative_to(PAYLOAD)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(PAYLOAD.rglob('*')) if path.is_file() and path.name!='package_files.json' and '__pycache__' not in path.parts}
    write(PAYLOAD/'package_files.json',{'files':files})
    all_files=[PAYLOAD/rel for rel in files]+[PAYLOAD/'package_files.json']
    bundle_dir=OUT/'bundle';bundle_dir.mkdir(exist_ok=True);archive=bundle_dir/'payload.tar.gz'
    with archive.open('wb') as rawstream,gzip.GzipFile(fileobj=rawstream,mode='wb',mtime=0,filename='') as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
        for path in sorted(all_files):
            info=tarfile.TarInfo(str(path.relative_to(PAYLOAD)));info.size=path.stat().st_size;info.mtime=0;info.mode=0o644
            with path.open('rb') as source:tar.addfile(info,source)
    if archive.stat().st_size>plan['bundle_transfer_budget_bytes']:raise RuntimeError('payload exceeds conservative transfer reserve')
    parts=[]
    with archive.open('rb') as stream:
        index=0
        while data:=stream.read(8*1024**2):
            path=bundle_dir/('part-%03d.bin'%index);path.write_bytes(data);parts.append({'path':str(path.relative_to(OUT)),'sha256':sha(path),'bytes':len(data)});index+=1
    bundle={'path':str(archive.relative_to(OUT)),'bytes':archive.stat().st_size,'sha256':sha(archive),'parts':parts}
    frozen_files={str(path.relative_to(OUT)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(OUT.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and not set(path.relative_to(OUT).parts)&{'run','reports','sources'} and path.name not in ('FREEZE.json','authorization.template.json','README_ZH.md')}
    freeze={'status':'PENDING_REVIEW_AND_EXPLICIT_GO','files':frozen_files,'bundle':bundle,'payload_bytes':sum(p.stat().st_size for p in all_files),'package_manifest_sha256':sha(PAYLOAD/'package_files.json'),'fixture_pixel_sha256':sha(fixture_dir/'texture.rgb'),'execution_contract_sha256':sha(PAYLOAD/'execution_contract.json'),'prior_compute_wall_seconds':previous,'new_model_forwards':0,'new_allocations':0,'formal_queries_executed':0}
    write(OUT/'FREEZE.json',freeze)
    write(OUT/'authorization.template.json',{'allocation_allowed':False,'freeze_sha256':sha(OUT/'FREEZE.json'),'scope':'one_T4_precheck_then_conditional_492','max_allocations':1,'note':'Template only; parent explicit GO required. Do not auto-enable.'})
    print(json.dumps({'freeze_sha256':sha(OUT/'FREEZE.json'),'bundle':{k:v for k,v in bundle.items() if k!='parts'},'parts':len(parts),'payload_bytes':freeze['payload_bytes'],'fixture_pixel_sha256':freeze['fixture_pixel_sha256'],'frozen_files':len(frozen_files)},indent=2))

if __name__=='__main__':main()
