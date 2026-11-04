"""Cumulative monotonic supervisor. No query starts before verified joint seal."""
import subprocess,sys,time,traceback
from datetime import datetime,timezone
from contracts import *

def main():
    original_start=datetime.fromisoformat('2026-11-02T05:36:42+00:00').timestamp()
    deadline=min(original_start+5400,datetime.fromisoformat('2026-11-02T10:00:00+00:00').timestamp())
    prior=read(PREFLIGHT/'TASK_RECEIPT.json');projection=read(PREFLIGHT/'RESOURCE_PROJECTION.json')
    assert sha(PREFLIGHT/'MANIFEST_SHA256.json')=='7198fbf0ec29a38c21dcba60b35128851b3f318303919af10eca2fc5c7a3b8d4'
    for item in read(PREFLIGHT/'MANIFEST_SHA256.json')['files']:assert sha(PREFLIGHT/item['path'])==item['sha256']
    assert sha(PREFLIGHT/'PROTOCOL.json')==prior['protocol_sha256']
    assert sha(PROJECT/'research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy')==read(PREFLIGHT/'ASSET_AUDIT.json')['npy_sha256']
    # The original600-second fixed allowance contained setup/integration and finalization.
    # Setup is now complete and fully charged by the uninterrupted cumulative clock.
    # Retain300seconds for final rechecking, persistence/reporting and release.
    remaining_projection=1.5*projection['numeric_and_io_seconds_before_margin']+300
    assert time.time()+remaining_projection<deadline,'Initial full remaining-cost gate failed'
    assert read(ROOT/'TEST_RESULTS.json')['status']=='PASS'
    source={'dinov2_small':'augmentation-selection-pilot-20261102','deit_tiny':'augmentation-selection-deit-20261102'}
    limits={'start_epoch':time.time(),'original_preparation_start_epoch':original_start,'deadline_epoch':deadline,'protocol_sha256':prior['protocol_sha256'],
            'identity_sha256':sha(PREFLIGHT/'IDENTITIES.json'),'preflight_bytes':sum(p.stat().st_size for p in PREFLIGHT.rglob('*') if p.is_file()),
            'common_sha256':{m:sha(PROJECT/d/'common.py') for m,d in source.items()},'margin_multiplier':1.5,
            'remaining_fixed_reserve_seconds':300,'completed_setup_charge_seconds':time.time()-original_start,'initial_remaining_projection_seconds':remaining_projection,
            'cumulative_budget_reset':False,'policy_note':'90min continuous from05:36:42UTC, includes prep, parent wait and integration; monotonic countdown in every process'}
    save(ROOT/'LIMITS.json',limits)
    code={p.name:sha(p) for p in ROOT.glob('*.py')};save(ROOT/'CODE_FREEZE.json',{'code':code,'protocol_sha256':prior['protocol_sha256'],'identity_sha256':limits['identity_sha256'],'query_utility_observed':False,'utc_epoch':time.time()})
    guard=Guard();receipts=[]
    def launch(phase,model):
        guard.check();print('START',phase,model,'remaining',round(guard.remaining()),flush=True)
        with (ROOT/(model+'_'+phase+'.stdout.log')).open('w') as out,(ROOT/(model+'_'+phase+'.stderr.log')).open('w') as err:
            child=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),phase,model],stdout=out,stderr=err)
            try:
                while child.poll() is None:
                    guard.check();time.sleep(2)
                if child.returncode:raise RuntimeError(f'{model} {phase} exited{child.returncode}; see stderr')
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        receipts.append(read(ROOT/model/(phase.upper()+'_RECEIPT.json')))
        print('DONE',phase,model,'remaining',round(guard.remaining()),flush=True)
    for model in MODELS:launch('fit',model)
    seals=validate_joint_gate(ROOT,limits['protocol_sha256'],limits['identity_sha256'])
    # Use actual support feature wall times to bound remaining query work, without query utility.
    N=8751;Q=9990
    empirical=sum(r['feature_seconds']/(N*9)*Q+read(PREFLIGHT/(r['encoder']+'_BENCHMARK.json'))['transform_seconds_8_originals']/8*Q for r in receipts)
    # Actual inclusive seconds/view, plus an extra conservative9-view transform charge per clean query.
    query_projection=1.5*(empirical+projection['io_seconds']+30)+300
    save(ROOT/'PREQUERY_BUDGET_GATE.json',{'remaining_seconds':guard.remaining(),'projected_query_and_finalize_seconds':query_projection,'passed':query_projection<guard.remaining(),'query_utility_observed':False,'actual_fit_receipts':receipts})
    assert query_projection<guard.remaining(),'Measured prequery cost gate failed; no query utility read'
    save(ROOT/'JOINT_QUERY_GATE.json',{'status':'BOTH_ENCODERS_SEALED','fit_seal_sha256':seals,'head_count':720,'choice_count':48,'utc_epoch':time.time(),'protocol_sha256':limits['protocol_sha256'],'identity_sha256':limits['identity_sha256']})
    for model in MODELS:launch('query',model)
    for model in MODELS:assert read(ROOT/model/'QUERY_SEAL.json')['status']=='ALL_QUERY_PREDICTIONS_SAVED'
    result=subprocess.run([sys.executable,str(ROOT/'analyze_results.py')],timeout=max(1,guard.remaining()-150),capture_output=True,text=True)
    (ROOT/'analysis.stdout.log').write_text(result.stdout);(ROOT/'analysis.stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    guard.check(reserve=120)
    save(ROOT/'RUN_RECEIPT.json',{'status':'COMPLETE','completed_utc':datetime.now(timezone.utc).isoformat(),'cumulative_wall_seconds':time.time()-original_start,'supervisor_receipts':receipts,'remaining_seconds':guard.remaining(),'all720_heads_and48_choices_sealed_before_queries':True,'new_CU':0,'model_processes_exited':True})
    print('COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(ROOT/'STOP_RECEIPT.json',{'status':'INCOMPLETE_STOPPED','error':repr(error),'traceback':traceback.format_exc(),'utc_epoch':time.time(),'query_started':any((ROOT/m/'query_features.inprogress.npy').exists() or (ROOT/m/'query_features.npy').exists() for m in MODELS)})
        raise
