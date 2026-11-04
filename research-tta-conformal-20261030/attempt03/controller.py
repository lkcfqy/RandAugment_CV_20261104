"""One authorized allocation, foreground work, verified durable local commits."""
import datetime,hashlib,json,os,re,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
from state_io import atomic_json,fsync_dir,sha_file,validate_checkpoint

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent
CLI=['/Users/lkc/.local/share/google-colab-cli/bin/python',str(OLD/'colab_guard.py')]
SESSION='tta-conformal-final-20261030'
REMOTE='/content/tta_attempt03'
PREVIOUS=1652.7684237957
DEADLINE=1793404407.1513782
ALLOCATION=None
RELEASED=threading.Event()
LOCK=threading.Lock()
IDENTITY=None
CONFIG=None
COUNTER=0
IO_TIMES=[]
LOGIT_IO=[]
for d in ('cli-ledger','commands','commits','results'): (ROOT/d).mkdir(exist_ok=True)
SPLIT=np.load(OLD/'partial-evidence/split-ids.npz')
HELD=np.concatenate([SPLIT[k] for k in ('pseudocal','validation','cal','eval')])
ROLES=np.concatenate([np.full(len(SPLIT[k]),i,dtype=np.int8) for i,k in enumerate(('pseudocal','validation','cal','eval'))])

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def clean(s):
    s=re.sub(r'ya29\.[A-Za-z0-9_.-]+','[REDACTED]',s)
    s=re.sub(r'(?i)(access_token|refresh_token|id_token|client_secret|authorization_code|runtime_proxy_token|colab-runtime-proxy-token)([\s\"\x27:=]+)[^\s,\"\x27&]+',r'\1\2[REDACTED]',s)
    s=re.sub(r'https://accounts\.google\.com/[^\s]+','[OAUTH URL OMITTED]',s)
    return re.sub(r'(?i)Bearer\s+[A-Za-z0-9_.-]+','Bearer [REDACTED]',s)

def status(phase,**kw):
    obj={'utc':now(),'phase':phase,'session':SESSION,'allocated_epoch':ALLOCATION,
         'previous_GPU_seconds':PREVIOUS,'wall_deadline':DEADLINE,**kw}
    atomic_json(ROOT/'STATUS.json',obj)
    with (ROOT/'controller-events.jsonl').open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def cli(label,args,timeout=150,required=True):
    global COUNTER
    with LOCK:COUNTER+=1;name=f'{COUNTER:04d}-{label}'
    t=time.time();env=os.environ.copy();env['COLAB_CLI_ASSIGN_RETRIES']='0'
    try:
        r=subprocess.run(CLI+args,capture_output=True,text=True,timeout=timeout,env=env)
        code=r.returncode;stdout=clean(r.stdout);stderr=clean(r.stderr)
    except subprocess.TimeoutExpired as e:
        code=124;stdout=clean((e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or ''))
        stderr='CLI timeout; remote outcome uncertain. No compute replay.'
    rec={'started_epoch':t,'finished_epoch':time.time(),'args':args,'exit_code':code,
         'stdout':stdout,'stderr':stderr,'new_oauth_enabled':False,'assignment_retries':0}
    atomic_json(ROOT/'cli-ledger'/(name+'.json'),rec)
    if required and code:raise RuntimeError(f'{label}: official CLI exit {code}; see {name}.json')
    return rec

def usage(label):
    rec=cli(label,['usage']);s=rec['stdout']
    return {'balance':float(re.search(r'Current balance: ([0-9.]+)',s)[1]),
            'rate':float(re.search(r'Usage rate: ([0-9.]+)/hr',s)[1]),
            'assignments':int(re.search(r'Active assignments: (\d+)',s)[1]),'receipt':rec}

def download(name,dest=None):
    p=dest or ROOT/'commits'/name;p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),'Refusing to overwrite committed '+str(p)
    part=p.with_name(p.name+'.part')
    rec=cli('get-'+name,['download','--session',SESSION,REMOTE+'/output/'+name,str(part)],required=False)
    if rec['exit_code']:
        # One read-only reconciliation; never resubmit training or allocate again.
        rec=cli('reconcile-get-'+name,['download','--session',SESSION,REMOTE+'/output/'+name,str(part)])
    assert part.exists() and part.stat().st_size>0
    with part.open('rb') as f:os.fsync(f.fileno())
    os.replace(part,p);fsync_dir(p.parent)
    return p

def json_download(name):return json.loads(download(name).read_text())

def execute(name,code,timeout=600):
    p=ROOT/'commands'/(name+'.py');assert not p.exists();p.write_text(code+'\n')
    r=cli('exec-'+name,['exec','--session',SESSION,'--file',str(p),'--timeout',str(timeout)],timeout+60,required=False)
    # A zero CLI exit does not prove that Python succeeded. An immutable done file is required.
    marker_start=time.perf_counter()
    # Reconcile an interrupted foreground call by its durable marker only.
    # No compute resubmission; all waiting remains within the cumulative budget.
    reconcile_deadline=min(time.time()+90,ALLOCATION+7050-PREVIOUS,DEADLINE-3600)
    while True:
        try:
            done=json_download(name+'.done.json');break
        except RuntimeError:
            if time.time()+15>=reconcile_deadline:raise
            time.sleep(15)
    r['done_marker_download_seconds']=time.perf_counter()-marker_start
    assert done['action']==name and done['success']
    if IDENTITY is not None:assert done['identity']==IDENTITY
    return done,r

def checkpoint(receipt):
    t=time.perf_counter();p=download(receipt['filename'])
    validate_checkpoint(p,receipt,IDENTITY)
    with p.open('rb') as f:os.fsync(f.fileno())
    atomic_json(p.with_suffix('.receipt.json'),receipt)
    seconds=time.perf_counter()-t
    IO_TIMES.append(seconds+receipt['save_seconds'])
    atomic_json(p.with_suffix('.local-commit.json'),{'utc':now(),'local_path':str(p),'remote_receipt':receipt,
                'sha256':sha_file(p),'full_state_readback':True,'durable_fsync':True,'download_validate_seconds':seconds})
    return p

def validate_logits(p,r):
    assert r['identity']==IDENTITY and p.stat().st_size==r['bytes'] and sha_file(p)==r['sha256']
    a=np.load(p,mmap_mode='r');lo,hi=r['row_start'],r['row_end']
    assert a.dtype.str==r['dtype']=='<f4' and list(a.shape)==r['shape']==[hi-lo,10,100] and np.isfinite(a).all()
    assert r['image_ids']==HELD[lo:hi].tolist() and r['roles']==ROLES[lo:hi].tolist()
    assert r['source']==['official_train' if i<50000 else 'official_test' for i in HELD[lo:hi]]
    assert r['source_indices']==[int(i if i<50000 else i-50000) for i in HELD[lo:hi]]
    assert r['view_ids']==list(range(10)) and r['class_ids']==list(range(100))
    assert r['view_definition']==[[0,0,False],[0,8,False],[8,0,False],[8,8,False],[4,4,False],
           [0,0,True],[0,8,True],[8,0,True],[8,8,True],[4,4,True]]
    assert hashlib.sha256(a.tobytes()).hexdigest()==r['logits_bytes_sha256']
    if not r['benchmark']:assert r['checkpoint_sha256']==json.loads((ROOT/'commits/formal080.receipt.json').read_text())['sha256']

def logits(r):
    t=time.perf_counter();p=download(r['filename']);validate_logits(p,r)
    atomic_json(p.with_suffix('.receipt.json'),r)
    seconds=time.perf_counter()-t;LOGIT_IO.append(seconds)
    atomic_json(p.with_suffix('.local-commit.json'),{'utc':now(),'sha256':sha_file(p),'identity_verified':True,'durable_fsync':True,'seconds':seconds})
    return p

def release(reason):
    if RELEASED.is_set():return
    rec=cli('release',['stop','--session',SESSION],120,required=False)
    ss=cli('released-sessions',['sessions'],120,required=False)
    u=usage('released-usage')
    verified=u['assignments']==0 and u['rate']==0 and SESSION not in ss['stdout']
    record={'utc':now(),'reason':reason,'stop_exit_code':rec['exit_code'],'sessions_stdout':ss['stdout'],
            'balance':u['balance'],'rate':u['rate'],'active_assignments':u['assignments'],'verified_zero':verified,
            'attempt_GPU_seconds_upper':time.time()-ALLOCATION if ALLOCATION else 0,
            'cumulative_GPU_seconds_upper':PREVIOUS+(time.time()-ALLOCATION if ALLOCATION else 0),
            'cumulative_CU_balance_difference_rounded':41.94-u['balance']}
    atomic_json(ROOT/'released.json',record)
    if verified:RELEASED.set()
    else:raise RuntimeError('Runtime release outcome not verified; explicit cleanup required')

def deadline_watch():
    while not RELEASED.wait(10):
        if PREVIOUS+time.time()-ALLOCATION>=7050 or time.time()>=DEADLINE-3600:
            try:release('hard-budget release reserve (117.5min cumulative;120min cap)')
            except Exception as e:atomic_json(ROOT/'cleanup-blocker.json',{'utc':now(),'error':str(e)})
            return

def preflight():
    assert not (ROOT/'allocation-attempted.json').exists(),'The sole recovery allocation is already spent or uncertain'
    assert sha_file(OLD/'PROTOCOL_ZH.md')=='28a8f92ca9fdd76cbcebe4998e9ff55f259f5e774a29dca190232f2eb5587f0b'
    assert sha_file(OLD/'protocol-lock.json')=='b25d20942cd32041a922eff50502e171253474fcb002968326478a9e7dcb690e'
    assert sha_file(OLD/'remote_experiment.py')=='cf56d1abef3067b5e54125b403233e2b97ae7973432080a54804111bb4fa42bf'
    assert json.loads((ROOT/'AMENDMENT_APPROVED.json').read_text())['status']=='APPROVED_FINAL_FORMAL_RECOVERY_105MIN_START_GATE'
    assert json.loads((OLD/'attempt02/post-release-durability-acceptance.json').read_text())['passed']
    assert sha_file(OLD/'attempt02/commits/benchmark_restore.done.json')==json.loads((ROOT/'AMENDMENT_APPROVED.json').read_text())['accepted_restore_receipt_sha256']
    cost=json.loads((ROOT/'PREALLOCATION_COST.json').read_text())
    assert cost['cumulative_projection_seconds']<=6300 and cost['CU_projection_at_rate_cap']<=6
    assert time.time()+cost['remaining_projection_seconds']+3600<DEADLINE
    size=sum(p.stat().st_size for d in (OLD,Path('/Users/lkc/Documents/Codex/2026-10-31/task/tta-conformal')) for p in d.rglob('*') if p.is_file())
    assert size+2*512*1024**2<2*1024**3
    assert time.time()<DEADLINE-7200
    recovered=(OLD/'results/run-state.json')
    if recovered.exists() and json.loads(recovered.read_text()).get('result')=='LOGITS_COMPLETE':
        raise RuntimeError('First-attempt complete result found; recovery allocation cancelled pending full validation')
    cli('preflight-sessions',['sessions']);u=usage('preflight-usage')
    assert u['assignments']==0 and u['rate']==0,'Existing assignment prevents exclusive rate attribution'
    status('PREFLIGHT_PASSED_NO_GPU',balance=u['balance'])

def main():
    global ALLOCATION,IDENTITY,CONFIG
    preflight()
    ALLOCATION=time.time()
    atomic_json(ROOT/'allocation-attempted.json',{'utc':now(),'epoch':ALLOCATION,'session':SESSION,'additional_allocation_number':2,'total_allocation_number':3,'no_further_allocation_authorized':True,'startup_gate_minutes':105,'running_hard_cap_minutes':120})
    try:
        cli('allocate',['new','--gpu','T4','--session',SESSION])
        st=cli('allocated-status',['status','--session',SESSION]);u=usage('allocated-usage')
        assert 'Hardware: T4' in st['stdout'] and 'Shape: Standard' in st['stdout']
        assert u['assignments']==1 and 0<=u['rate']<=3 and u['balance']>=6
        status('GPU_ALLOCATED_NO_NEW_BENCHMARK_AUTHORIZED',gpu='T4',verified_rate_cu_per_hour=u['rate'],balance=u['balance'])
        threading.Thread(target=deadline_watch,daemon=True).start()
        summary=json.loads((OLD/'partial-evidence/split-summary.json').read_text())
        CONFIG={'allocated_at':ALLOCATION,'previous_GPU_seconds':PREVIOUS,'verified_rate':u['rate'],'wall_deadline':DEADLINE,
          'protocol_sha256':sha_file(OLD/'PROTOCOL_ZH.md'),'amendment_sha256':sha_file(ROOT/'AMENDMENT_APPROVED.json'),
          'science_source_sha256':sha_file(OLD/'remote_experiment.py'),'dataset_sha256':summary['archive_sha256'],
          'split_manifest_sha256':sha_file(OLD/'partial-evidence/split-manifest.jsonl'),
          'role_id_hashes':{k:v['ids_sha256'] for k,v in summary['roles'].items()},
          'accepted_restore_receipt_sha256':sha_file(OLD/'attempt02/commits/benchmark_restore.done.json'),'new_acceptance_updates_allowed':0}
        atomic_json(ROOT/'runtime-config.json',CONFIG)
        setup=ROOT/'commands/mkdir.py';setup.write_text("from pathlib import Path\nPath('/content/tta_attempt03').mkdir(exist_ok=True)\n")
        cli('mkdir',['exec','--session',SESSION,'--file',str(setup),'--timeout','45'],105)
        for local,remote in ((OLD/'remote_experiment.py','science_v1.py'),(ROOT/'state_io.py','state_io.py'),(ROOT/'attempt03_engine.py','attempt03_engine.py')):
            cli('upload-'+remote,['upload','--session',SESSION,str(local),REMOTE+'/'+remote])
        prepare,prepcli=execute('prepare',"import sys\nsys.path.insert(0,'/content/tta_attempt03')\nfrom attempt03_engine import Engine\n_tta=Engine("+repr(CONFIG)+")")
        IDENTITY=prepare['identity']
        assert IDENTITY['engine_sha256']==sha_file(ROOT/'attempt03_engine.py') and IDENTITY['state_io_sha256']==sha_file(ROOT/'state_io.py')
        assert IDENTITY['protocol_sha256']==CONFIG['protocol_sha256'] and IDENTITY['amendment_sha256']==CONFIG['amendment_sha256']
        status('DATA_IDENTITY_VERIFIED',identity=IDENTITY)
        # Reuse the accepted physical200-step reference/replay test from attempt02. No extra SGD or views.
        locked=json.loads((ROOT/'PREALLOCATION_COST.json').read_text());measured=time.time()
        remaining=locked['remaining_after_preparation_seconds']
        projected=PREVIOUS+measured-ALLOCATION+remaining
        prior_CU=PREVIOUS/3600*1.15
        gate={'measured_epoch':measured,'utc':now(),'prior_GPU_seconds':PREVIOUS,
              'actual_new_allocation_elapsed_seconds':measured-ALLOCATION,
              'conservative_remaining_seconds':remaining,'cumulative_projection_seconds':projected,
              'cumulative_projection_CU':prior_CU+(measured-ALLOCATION+remaining)/3600*u['rate'],
              'locked_cost_sha256':sha_file(ROOT/'PREALLOCATION_COST.json'),'startup_gate_seconds':6300,
              'running_hard_cap_seconds':7200,'new_benchmark_updates':0,'new_benchmark_views':0,
              'passed':projected<=6300 and prior_CU+(measured-ALLOCATION+remaining)/3600*u['rate']<=6 and measured+remaining+3600<DEADLINE}
        atomic_json(ROOT/'performance-gate.json',gate)
        status('ACTUAL_PREPARATION_COST_GATE',gate=gate,formal_training_started=False)
        if not gate['passed']:raise RuntimeError('RESOURCE_STOP_PERFORMANCE: current full cumulative projection exceeds approved105min startup gate')
        started,_=execute('formal_start','_result=_tta.start_formal('+repr(gate)+')')
        previous=started['checkpoint'];checkpoint(previous)
        status('FORMAL_TRAINING_STARTED',epoch=0,checkpoint_sha256=previous['sha256'])
        for target in range(5,81,5):
            assert not RELEASED.is_set()
            done,_=execute(f'train{target:03d}',f"_result=_tta.train_chunk({Path(previous['filename']).stem!r},{previous['sha256']!r},{target})")
            previous=done['checkpoint'];checkpoint(previous)
            status('FORMAL_CHECKPOINT_DURABLE',epoch=target,steps=target*313,checkpoint_sha256=previous['sha256'],audit=done['audit'])
        for index in range(40):
            done,_=execute(f'infer{index:03d}',f'_result=_tta.infer_chunk({index})')
            logits(done['logits'])
            status('LOGIT_SHARD_DURABLE',shard=index,rows=min((index+1)*512,20000),audit=done['audit'])
        assert done['audit']=={'benchmark_steps':0,'benchmark_presentations':0,'formal_steps':25040,'formal_presentations':3200000,'inference_views':200000}
        atomic_json(ROOT/'results/run-state.json',{'result':'LOGITS_COMPLETE',**done['audit'],
          'benchmark_steps':200,'benchmark_presentations':25600,'inference_views':205120,
          'audit_scope':'Formal trajectory plus already-completed accepted qualification from attempt02; no new qualification updates/views.',
          'current_allocation_audit':done['audit'],'prior_qualification_audit':json.loads((OLD/'attempt02/commits/benchmark_restore.done.json').read_text())['audit'],
          'identity':IDENTITY})
        for name in ('heldout-image-ids.npy','heldout-labels.npy','view-ids.npy','split-ids.npz','split-manifest.jsonl','split-summary.json','normalization.json','environment.json','events.jsonl'):
            download(name,ROOT/'results'/name)
        assert sha_file(ROOT/'results/split-manifest.jsonl')==CONFIG['split_manifest_sha256']
        assert np.array_equal(np.load(ROOT/'results/heldout-image-ids.npy'),HELD)
        release('Full epoch80 checkpoint and all40 formal logit shards verified locally')
        assemble()
        status('GPU_RELEASED_LOGITS_COMPLETE_READY_FOR_CPU_ANALYSIS',released=json.loads((ROOT/'released.json').read_text()))
    except BaseException as e:
        error={'utc':now(),'result':'INFRASTRUCTURE_OR_RESOURCE_BLOCKED_NOT_SCIENTIFIC_NEGATIVE','error':str(e),
               'traceback':traceback.format_exc(),'formal_status':json.loads((ROOT/'STATUS.json').read_text()) if (ROOT/'STATUS.json').exists() else None}
        atomic_json(ROOT/'BLOCKER.json',error);status('BLOCKED_RELEASE_REQUIRED',error=str(e))
        # Read-only evidence capture is bounded. No extra allocation or training replay.
        for name in ('events.jsonl',):
            if not (ROOT/'commits'/('failure-'+name)).exists():
                try:download(name,ROOT/'commits'/('failure-'+name))
                except Exception:pass
        if not RELEASED.is_set():
            try:release('exception: '+str(e))
            except Exception as ce:atomic_json(ROOT/'cleanup-blocker.json',{'utc':now(),'error':str(ce)})
        raise

def assemble():
    data=ROOT/'results';dest=data/'logits-fp32.npy'
    out=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(20000,10,100))
    for i in range(40):
        p=ROOT/'commits'/f'infer{i:03d}.npy';r=json.loads(p.with_suffix('.receipt.json').read_text());validate_logits(p,r)
        out[r['row_start']:r['row_end']]=np.load(p,mmap_mode='r')
    out.flush();del out
    read=np.load(dest,mmap_mode='r')
    np.save(data/'baseline-logits-fp32.npy',read[:,4,:]);np.save(data/'mean-logits-f64.npy',read.mean(1,dtype=np.float64))
    assert np.array_equal(np.load(data/'baseline-logits-fp32.npy'),read[:,4,:])
    assert np.array_equal(np.load(data/'mean-logits-f64.npy'),read.mean(1,dtype=np.float64))
    atomic_json(data/'cache-acceptance.json',{'pass':True,'all40_shards_identity_and_hash_verified':True,'baseline_and_mean_readback_exact':True,
                'final_checkpoint_sha256':json.loads((ROOT/'commits/formal080.receipt.json').read_text())['sha256'],'logits_sha256':sha_file(dest)})

if __name__=='__main__':main()
