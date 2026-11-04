"""Local static/assets and guarded artificial tests; never construct a model."""
import ast,contextlib,hashlib,io,json,os,socket,sys,time,unittest
from pathlib import Path

BASE=Path(__file__).resolve().parent
PAYLOAD=BASE/'payload'
sys.path.insert(0,str(PAYLOAD/'scripts'));sys.path.insert(0,str(BASE/'tests'))
from contracts import canonical,validate_query

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    begin=time.monotonic();report={};counters={'module_forward_attempts':0,'network_attempts':0,'pickle_attempts':0,'GPU_API_attempts':0}
    def deny(key):
        def fail(*a,**kw):counters[key]+=1;raise AssertionError(key+' forbidden in code/asset verification')
        return fail
    socket.socket.connect=deny('network_attempts');socket.socket.connect_ex=deny('network_attempts');socket.create_connection=deny('network_attempts')
    import torch
    torch.nn.Module._call_impl=deny('module_forward_attempts');torch.load=deny('pickle_attempts')
    for key in ('is_available','synchronize','get_device_name','set_per_process_memory_fraction','empty_cache'):setattr(torch.cuda,key,deny('GPU_API_attempts'))
    for path in (PAYLOAD/'scripts').glob('*.py'):ast.parse(path.read_text())
    suite=unittest.defaultTestLoader.discover(str(BASE/'tests'),pattern='test_*.py')
    output=io.StringIO()
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    report['tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful()}
    (BASE/'reports').mkdir(exist_ok=True);(BASE/'reports/verification_tests.log').write_text(output.getvalue())
    selection=json.loads((PAYLOAD/'science/selection_frozen.json').read_text());labels=json.loads((PAYLOAD/'science/labels_frozen.json').read_text());queries=json.loads((PAYLOAD/'science/queries_frozen.json').read_text())['queries']
    source={x['filename']:x for x in selection['images']};names={x['pair_id']:x for x in labels['pairs']};images={}
    for q in queries:
        validate_query(q,source[q['filename']],names[q['pair_id']])
        images.setdefault(q['filename'],sha(PAYLOAD/'science/images'/q['filename']))
        if images[q['filename']]!=q['image_sha256']:raise ValueError('query image bytes mismatch')
    if len(queries)!=492 or len(images)!=82 or len(names)!=47 or sum(row['eligible'] for row in names.values())!=41:raise ValueError('fixed denominator mismatch')
    report['metadata_only_input_validation']={'query_contracts':len(queries),'eligible_images_hashed':len(images),'all_pairs':len(names),'eligible_pairs':sum(x['eligible'] for x in names.values()),'truth_bound_to_labels':True,'images_decoded':0}
    fixture=json.loads((PAYLOAD/'fixture/manifest.json').read_text());report['fixture']=fixture
    if sha(PAYLOAD/'fixture/texture.rgb')!=fixture['pixel_sha256']:raise ValueError('fixture bytes changed')
    protocol=json.loads((PAYLOAD/'execution_contract.json').read_text());plan=json.loads((PAYLOAD/'plan.json').read_text())
    if protocol['gate_pair_condition']['C_operator']!='>=' or protocol['gate_pair_condition']['V_operator']!='>=':raise ValueError('inclusive boundary missing')
    if protocol['metric_code_sha256']!=sha(PAYLOAD/'scripts/metrics.py') or plan['previous_compute_wall_seconds']!=259.6864732820104:raise ValueError('metric/budget freeze mismatch')
    if sha(BASE.parent/'observation-entry/protocol_v2_frozen.json')!='7f8975bd2150de7d27b6a6cca35b715efca051ee6e6b6e0eb94d632702ba5bc1':raise ValueError('historical protocol altered')
    old_base=BASE.parent/'cuda-runplan-20261031'
    if sha(old_base/'FREEZE.json')!='915d27011f9f948150de676b08f5830d5811066b993fc500aad729f0f284849a':raise ValueError('old freeze changed')
    if plan['prior_credit_upper_bound_CU']!=.14396 or plan['download_cap_bytes']!=5*1024**3 or plan['cumulative_transfer_forecast_bytes']>plan['download_cap_bytes']:raise ValueError('recovery resource ledger differs')
    old_files=json.loads((old_base/'payload/package_files.json').read_text())['files']
    new_files=json.loads((PAYLOAD/'package_files.json').read_text())['files']
    if set(old_files)!=set(new_files):raise ValueError('recovery added or removed package files')
    unchanged_prefixes=('source/','science/','fixture/','models/','manifests/','wheels/')
    unchanged_scripts={'scripts/metrics.py','scripts/numerics.py','scripts/model_io.py','scripts/fixture.py','scripts/hooks.py'}
    for rel in old_files:
        if rel.startswith(unchanged_prefixes) or rel in unchanged_scripts:
            if new_files[rel]!=old_files[rel]:raise ValueError('scientific identity changed: '+rel)
    report['package_roster_unchanged']=True;report['previous_freeze_preserved']=True
    expected={'selection_frozen.json':'c7ba6b400d0ec6123eda4274625c4d5a64111a664f978001e31215b3ad27b6bc','labels_frozen.json':'8f22518b77d7a5fe066503fee99b3250fbd83d5e42185f5caf2b4eef3e8e0ff2','queries_frozen.json':'29b68745766b2e1e594f0704a57be46ac098bc43aa3315cef53a85ababdbfab6'}
    for name,expected_sha in expected.items():
        if sha(PAYLOAD/'science'/name)!=expected_sha:raise ValueError('original frozen inputs changed')
    report['unchanged_original_inputs']=expected;report['protocol_version']=protocol['protocol_version'];report['counters']=counters
    report.update(status='PASS' if result.wasSuccessful() and not any(counters.values()) else 'FAIL',verification_wall_seconds=time.monotonic()-begin,new_allocations=0,new_uploads=0,new_forward_calls=0,formal_queries_executed=0,prior_compute_wall_seconds=259.6864732820104)
    (BASE/'reports/verification.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','tests','counters','metadata_only_input_validation','verification_wall_seconds')},indent=2))
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
