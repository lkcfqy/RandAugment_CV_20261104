"""Finite pilot controller. Invoke only through supervise.py; no automatic expansion."""
from core import *
import argparse,importlib.metadata as metadata,gc,traceback
from datetime import datetime,timezone

def status(stage,**extra):
    p=ROOT/'STATUS.json';s=json.loads(p.read_text())
    s.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra);dump(p,s)
    (ROOT/'STATUS_ZH.md').write_text('# 实时进度\n\n'+s['updated_utc']+'\n\n阶段：'+stage+'。\n\n'+json.dumps(extra,ensure_ascii=False,indent=2)+'\n\n协议不变。未执行不代表科学阴性。详情见 STATUS.json、mps-ledger.json 和 logs/。\n')
    print(json.dumps({'stage':stage,**extra},ensure_ascii=False),flush=True)
def used_seconds():
    ledger=json.loads((BASE/'mps-ledger.json').read_text())
    return ledger['cumulative_seconds']+(time.time()-ledger['active']['started_epoch'] if ledger.get('active') else 0)
def sources():
    paths=['core.py','pilot.py','oracle.py','supervise.py','manifest.py','PROTOCOL_ZH.md','dependency-lock.json','AMENDMENT.json','splits.json','data-manifest.json']
    result={p:sha(ROOT/p) for p in paths}
    import inspect
    for obj in [timm.models.vision_transformer.VisionTransformer,T.RandAugment,T.RandomResizedCrop,T.RandomHorizontalFlip]:
        path=Path(inspect.getfile(obj));result[str(path)]=sha(path)
    return result
def cleanup(m,opt=None):
    del m,opt;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
def initial_path(seed):return ROOT/'initial'/f'seed{seed}.pt'
def run_path(seed,recipe):return ROOT/'runs'/f'seed{seed}-{recipe}'
def kind_for(recipe):return {'B1':'weak','B2':'dual','B3':'dual','B4':'ent','B5':'weak','B6':'random','C1':'candidate','C2':'candidate'}[recipe]

def linear_probes(data):
    status('LP_support_features',training_steps=0,formal_runs_completed=0)
    allids=sorted(set().union(*map(set,data.splits['support'].values())))
    m=model();m.eval().requires_grad_(False);features_cpu=[];started=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(allids),32):
            x,_=data.eval_batch(allids[i:i+32]);features_cpu.append(features(m,x).cpu())
    f=torch.cat(features_cpu);torch.mps.synchronize();elapsed=time.perf_counter()-started
    save_pt(ROOT/'initial'/'support_features.pt',{'ids':allids,'features':f,'splits_sha256':sha(ROOT/'splits.json')})
    indices={ident:i for i,ident in enumerate(allids)};rows=[];del m;gc.collect()
    for seed in [11,22,33]:
        began=time.perf_counter();m=model(seed);m.eval().requires_grad_(False);m.head.requires_grad_(True)
        pre=tensorhash(m.head.state_dict());ids=data.splits['support'][str(seed)]
        x=f[[indices[i] for i in ids]].to(DEVICE);y=torch.tensor([data.rows[i]['class_id'] for i in ids],device=DEVICE)
        opt=optimizer(m,lp=True)
        for step in range(200):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m.head(x),y);loss.backward();opt.step()
        torch.mps.synchronize();assert torch.isfinite(loss)
        init_hash=tensorhash(m.state_dict())
        digest=save_pt(initial_path(seed),{'model':m.state_dict(),'seed':seed,'head_pre_lp_state_sha256':pre,'initial_state_sha256':init_hash,'lp_updates':200,'splits_sha256':sha(ROOT/'splits.json')})
        rows.append({'seed':seed,'lp_updates':200,'head_pre_lp_state_sha256':pre,'initial_state_sha256':init_hash,'file_sha256':digest,'seconds':time.perf_counter()-began})
        status('LP_complete_seed',seed=seed,lp_seeds_completed=len(rows),formal_runs_completed=0)
        del m,opt,x,y;gc.collect();torch.mps.empty_cache()
    dump(ROOT/'lp.json',{'status':'complete','support_feature_images':len(allids),'support_features_seconds':elapsed,'seeds':rows,'optimizer_inherited':False})

def calibration(data):
    specs=[('weak','weak',10**9),('dual','dual',0),('ent_weak','ent',10**9),('ent_active','ent',0),('candidate','candidate',0),('random','random',0),('head','head',0)]
    output={};load_times=[];save_times=[]
    for name,kind,warm in specs:
        status('throughput_calibration',path=name,throughput_calibration_started=True,formal_runs_completed=0)
        load=time.perf_counter();m=model(11,initial_path(11));load_times.append(time.perf_counter()-load)
        m.train(kind!='head')
        if kind=='head':
            for n,p in m.named_parameters():p.requires_grad_(n.startswith('head.'))
        opt=optimizer(m,head_only=kind=='head')
        cache={i:(key('calcache',i)%10)/9 if name=='ent_active' else 0. for i in data.splits['support']['11']}
        for u in range(5):update(m,opt,data,11,u,kind,cache,warm,u+1,1000)
        torch.mps.synchronize();blocks=[]
        for block in range(3):
            started=time.perf_counter()
            for j in range(10):
                u=5+block*10+j;update(m,opt,data,11,u,kind,cache,warm,u+1,1000)
            torch.mps.synchronize();blocks.append((time.perf_counter()-started)/10)
        output[name]={'warmup_updates':5,'timed_updates':30,'seconds_per_step_blocks':blocks,'cr':float(np.median(blocks)),'conservative_seconds_per_step':max(blocks)}
        if name in ['dual','head']:
            begun=time.perf_counter();temp=ROOT/'calibration-checkpoint.pt'
            save_pt(temp,{'model':m.state_dict(),'optimizer':opt.state_dict(),'cache':cache,'rng_cpu':torch.get_rng_state(),'rng_mps':torch.mps.get_rng_state()})
            save_times.append(time.perf_counter()-begun);output[name]['checkpoint_bytes']=temp.stat().st_size
            temp.unlink();temp.with_suffix('.pt.sha256.json').unlink()
        print(json.dumps({'path':name,**output[name]}),flush=True)
        dump(ROOT/'calibration-progress.json',output)
        del m,opt;gc.collect();torch.mps.empty_cache()
    # Measure deterministic decode/forward chain on support only, without accuracy.
    m=model(11,initial_path(11));m.eval();ids=data.splits['support']['11'];batches=len(ids)//32
    with torch.no_grad():
        for u in range(5):x,_=data.eval_batch(ids[(u%batches)*32:(u%batches+1)*32]);m(x)
        torch.mps.synchronize();blocks=[]
        for block in range(3):
            begun=time.perf_counter()
            for j in range(10):
                u=(block*10+j)%batches;x,_=data.eval_batch(ids[u*32:(u+1)*32]);m(x)
            torch.mps.synchronize();blocks.append((time.perf_counter()-begun)/10)
    output['evaluation_support_only']={'seconds_per_batch_blocks':blocks,'conservative_seconds_per_batch':max(blocks),'accuracy_computed':False}
    output['model_load_seconds_max']=max(load_times);output['checkpoint_seconds_max']=max(save_times)
    output['calibration_cache_fixture']='ent_active initial qi=key(calcache,image_id)%10/9; overwritten by observed confidence; disposable calibration only'
    dump(ROOT/'calibration.json',output);del m;gc.collect();torch.mps.empty_cache();return output

def preflight():
    assert json.loads((ROOT/'gradient-oracle.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'gradient-oracle.json').read_text())['core_sha256']==sha(ROOT/'core.py')
    assert json.loads((ROOT/'data-validation.json').read_text())['status']=='passed'
    assert not (ROOT/'protocol.lock.json').exists()
    data=Data();order=[{'seed':s,'recipe':r} for s in [11,22,33] for r in RECIPES];random.Random(20261030).shuffle(order)
    dump(ROOT/'run-order.json',{'shuffle_seed':20261030,'enumeration':'seeds [11,22,33], recipes B1..B6,C1,C2','order':order})
    linear_probes(data);c=calibration(data)
    costs={'B1':c['weak'],'B2':c['dual'],'B3':c['dual'],'B4':{
           'cr':max(c['ent_weak']['cr'],c['ent_active']['cr']),
           'conservative_seconds_per_step':max(c['ent_weak']['conservative_seconds_per_step'],c['ent_active']['conservative_seconds_per_step'])},
           'B6':c['random'],'C1':c['candidate'],'C2':c['candidate']}
    runs={r:{'N':math.floor(180/v['cr']),'kind':kind_for(r),'cr':v['cr'],'conservative_step_seconds':v['conservative_seconds_per_step']} for r,v in costs.items()}
    nft=math.floor(144/c['weak']['cr']);nh=math.floor(36/c['head']['cr'])
    runs['B5']={'N':nft+nh,'N_FT':nft,'N_head':nh,'kind':'weak_then_head','cr_FT':c['weak']['cr'],'cr_head':c['head']['cr']}
    failures=[]
    for r,v in runs.items():
        if r!='B5' and v['N']<128:failures.append(f'{r}: N={v["N"]}<128')
    if nft<96 or nh<16:failures.append(f'B5: FT={nft},head={nh}; requires 96+16')
    minimum=min(v['N'] for v in runs.values());choices=[j for j in [64,96,128,192,256,384,512] if j<=minimum]
    J=max(choices) if choices else None
    for r,v in runs.items():
        v['low_backbone_lr']=r in ['B3','C2']
        v['stage_schedules']=[{'N':n,'warmup':math.ceil(.05*n),'final_base_lr_fraction':.1} for n in ([v['N_FT'],v['N_head']] if r=='B5' else [v['N']])]
        v['projected_loop_seconds']=(nft*c['weak']['conservative_seconds_per_step']+nh*c['head']['conservative_seconds_per_step']) if r=='B5' else v['N']*v['conservative_step_seconds']
    training=3*sum(v['projected_loop_seconds'] for v in runs.values())
    batches=48*math.ceil(len(data.splits['validation'])/32)+6*math.ceil(len(data.splits['test'])/32)
    evaluation=batches*c['evaluation_support_only']['conservative_seconds_per_batch']
    overhead=24*(7*c['checkpoint_seconds_max']+c['model_load_seconds_max'])+6*c['model_load_seconds_max']+120
    future=training+evaluation+overhead;used=used_seconds()
    if used+future>7200:failures.append('Full 24-run plan exceeds remaining MPS stage time')
    if time.time()+future>datetime(2026,10,31,1,14,tzinfo=timezone.utc).timestamp():failures.append('Full plan exceeds wall deadline')
    actual=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file());weight_bytes=initial_path(11).stat().st_size
    disk=actual+(48+3)*weight_bytes+3*c['dual']['checkpoint_bytes']+200*1024**2
    if disk>3*1024**3:failures.append('Conservative complete artifact estimate exceeds 3GiB')
    projection={'mps_already_used':used,'training_seconds':training,'evaluation_seconds':evaluation,'overhead_and_reserve_seconds':overhead,
                'remaining_full_plan_seconds':future,'total_projected_mps_seconds':used+future,'disk_bytes_current':actual,'disk_bytes_peak_projected':disk,
                'state_file_bytes':weight_bytes,'run_count':24,'test_model_count':6}
    result={'status':'resource_undetermined' if failures else 'passed','failures':failures,'runs':runs,'J':J,'projection':projection}
    dump(ROOT/'preflight.json',result)
    if failures:status('resource_undetermined',resource_failures=failures,scientific_status='undetermined');raise RuntimeError('; '.join(failures))
    lock={'protocol_id':'cifar100_view_route_pilot_v1','locked_utc':datetime.now(timezone.utc).isoformat(),'implementation_hashes':sources(),
          'recipes':runs,'run_order':order,'J':J,'projection':projection,'versions':{'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,'numpy':np.__version__},
          'schedule':'for t=1..ceil(.05N): t/w; then .1+.9*(1+cos(pi*(t-w)/(N-w)))/2',
          'rng_key':'SHA256 UTF8 pets-route-v1|rng-v1|domain|seed|position... first8 big-endian & (2^63-1)',
          'checkpoint_policy':'quarters/J/stage/final, two alternating active slots, safe readback; completed slots removed only after permanent final/J outputs',
          'resume_limit':1,'test_selection':'global recipe by summed final-val integer correct over 3 seeds; tie ID; only 6 selected models',
          'usefulness_gate':{'mean_paired_test_pp_at_least':1.,'all_three_seed_delta_strictly_positive':True,'significance_claim':False}}
    dump(ROOT/'protocol.lock.json',lock);dump(ROOT/'protocol.lock.sha256.json',{'sha256':sha(ROOT/'protocol.lock.json')})
    status('protocol_locked',protocol_lock_created=True,formal_runs_completed=0,preflight_projection=projection,N={r:v['N'] for r,v in runs.items()},J=J)

def check_lock(lock):
    assert sha(ROOT/'protocol.lock.json')==json.loads((ROOT/'protocol.lock.sha256.json').read_text())['sha256']
    for name,digest in lock['implementation_hashes'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(p)==digest,('Implementation changed after lock',name)

def checkpoint(folder,sequence,m,opt,seed,recipe,u,stage,stage_n,cache,loop_seconds,run_started,lock):
    slot=folder/f'recovery-{sequence%2}.pt'
    payload={'model':m.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'recipe':recipe,'completed_updates':u,'next_global_update':u,
             'stage':stage,'scheduler':{'stage_N':stage_n,'completed_stage_updates':u if stage==1 else u-lock['recipes'][recipe]['N_FT']},
             'ent_cache':cache,'torch_cpu_rng':torch.get_rng_state(),'torch_mps_rng':torch.mps.get_rng_state(),'python_rng':random.getstate(),
             'augmentation_rng':'stateless hash keys; next_global_update fully determines data/weak/RA/routing positions',
             'data_epoch':u//50,'data_batch_within_epoch':u%50,'loop_seconds':loop_seconds,'run_elapsed_seconds':time.perf_counter()-run_started,
             'mps_cumulative_at_save':used_seconds(),'lock_sha256':sha(ROOT/'protocol.lock.json'),'checkpoint_sequence':sequence}
    digest=save_pt(slot,payload);dump(folder/'recovery-latest.json',{'slot':slot.name,'sha256':digest,'sequence':sequence,'completed_updates':u})

def restore_checkpoint(folder,lock):
    valid=[];errors=[]
    for p in sorted(folder.glob('recovery-*.pt')):
        try:
            receipt=json.loads(p.with_suffix('.pt.sha256.json').read_text())
            assert sha(p)==receipt['sha256']
            cp=torch.load(p,map_location='cpu',weights_only=True)
            assert cp['lock_sha256']==sha(ROOT/'protocol.lock.json')
            valid.append((cp['checkpoint_sequence'],p,cp))
        except Exception as exc:errors.append({'file':p.name,'error':str(exc)})
    if not valid:raise RuntimeError('VALIDATION: no verified recovery slot; no silent run restart')
    _,p,cp=max(valid,key=lambda x:x[0])
    dump(folder/'recovery-resumed.json',{'slot':p.name,'sha256':sha(p),'completed_updates':cp['completed_updates'],'rejected_slots':errors,'mps_used_before_resume':used_seconds()})
    return cp

def formal(resume=False):
    lock=json.loads((ROOT/'protocol.lock.json').read_text());check_lock(lock);data=Data();completed=[]
    if (ROOT/'results.json').exists():return
    existing=any((run_path(x['seed'],x['recipe'])/'started.json').exists() for x in lock['run_order'])
    assert not existing or resume,'Existing run detected; use one bounded explicit --resume, never silently rerun'
    if resume:
        assert existing and not (ROOT/'resume-receipt.json').exists(),'At most one recovery attempt'
        failure=ROOT/'formal-failure.json'
        if failure.exists():
            message=json.loads(failure.read_text())['message'].lower()
            assert not any(s in message for s in ['out of memory','not implemented','unsupported','resource:']),'No rescue of OOM, unsupported operator, or resource failure'
        dump(ROOT/'resume-receipt.json',{'attempt':1,'utc':datetime.now(timezone.utc).isoformat(),'mps_used_before_resume':used_seconds(),'protocol_unchanged':True})
    for index,item in enumerate(lock['run_order']):
        seed,recipe=item['seed'],item['recipe'];cfg=lock['recipes'][recipe];n=cfg['N'];folder=run_path(seed,recipe)
        if (folder/'complete.json').exists():
            record=json.loads((folder/'complete.json').read_text());assert sha(folder/'final.pt')==record['final_file_sha256'];completed.append(record);continue
        cp=restore_checkpoint(folder,lock) if (folder/'started.json').exists() else None
        start_u=cp['completed_updates'] if cp else 0
        remaining=lock['run_order'][index:]
        future=sum(lock['recipes'][x['recipe']]['projected_loop_seconds'] for x in remaining)
        future-=cfg['projected_loop_seconds']*start_u/n
        eval_each=2*math.ceil(len(data.splits['validation'])/32)
        cal=json.loads((ROOT/'calibration.json').read_text())
        future+=(len(remaining)*eval_each+6*math.ceil(len(data.splits['test'])/32))*cal['evaluation_support_only']['conservative_seconds_per_batch']
        future+=len(remaining)*(7*cal['checkpoint_seconds_max']+cal['model_load_seconds_max'])+6*cal['model_load_seconds_max']+60
        if used_seconds()+future>7200:raise RuntimeError('RESOURCE: actual remaining budget cannot complete all remaining runs and final evaluation')
        folder.mkdir(parents=True,exist_ok=True)
        began=time.perf_counter();m=model(seed,initial_path(seed));m.train();init=tensorhash(m.state_dict())
        expected=torch.load(initial_path(seed),map_location='cpu',weights_only=True)['initial_state_sha256'];assert init==expected
        opt=optimizer(m,low=cfg['low_backbone_lr']);cache={i:0. for i in data.splits['support'][str(seed)]};loop_seconds=0.;seq=0;stage=1
        record={'seed':seed,'recipe':recipe,'configuration':cfg,'initial_state_sha256':init,'initial_file_sha256':sha(initial_path(seed)),
                'splits_sha256':sha(ROOT/'splits.json'),'lock_sha256':sha(ROOT/'protocol.lock.json'),'started_utc':datetime.now(timezone.utc).isoformat(),'J':lock['J'],'mps_started_seconds':used_seconds()-(time.perf_counter()-began)}
        trace=[];jresult=None
        if cp:
            record=json.loads((folder/'started.json').read_text());assert record['initial_state_sha256']==init
            m.load_state_dict(cp['model'],strict=True);stage=cp['stage']
            if stage==2:
                m.eval()
                for name,p in m.named_parameters():p.requires_grad_(name.startswith('head.'))
                opt=optimizer(m,head_only=True)
            opt.load_state_dict(cp['optimizer']);cache=cp['ent_cache'];loop_seconds=cp['loop_seconds'];seq=cp['checkpoint_sequence'];began-=cp['run_elapsed_seconds']
            assert set(cache)==set(data.splits['support'][str(seed)])
            torch.set_rng_state(cp['torch_cpu_rng']);torch.mps.set_rng_state(cp['torch_mps_rng']);random.setstate(cp['python_rng'])
            if (folder/'training-trace.json').exists():trace=json.loads((folder/'training-trace.json').read_text())[:start_u]
            if start_u>=lock['J']:
                if not (folder/'J-validation.pt').exists():
                    assert start_u==lock['J'],'Missing historical J snapshot cannot be reconstructed from later weights'
                    save_pt(folder/'J.pt',{'model':m.state_dict(),'completed_updates':start_u,'stage':stage,'initial_state_sha256':init})
                    predict(m,data,data.splits['validation'],folder/'J-validation.pt')
                jp=torch.load(folder/'J-validation.pt',map_location='cpu',weights_only=True)
                js=torch.load(folder/'J.pt',map_location='cpu',weights_only=True)
                jresult={k:jp[k] for k in ['correct','count','seconds']};jresult['stage']=js['stage']
        else:dump(folder/'started.json',record)
        status('formal_training',seed=seed,recipe=recipe,completed_updates=start_u,N=n,formal_runs_completed=len(completed))
        points={math.ceil(n*k/4) for k in range(1,5)}|{lock['J']}
        if recipe=='B5':points.add(cfg['N_FT'])
        last_progress=time.monotonic()
        for u in range(start_u,n):
            if recipe=='B5' and u==cfg['N_FT']:
                save_pt(folder/'stage1-boundary.pt',{'model':m.state_dict(),'completed_updates':u,'stage':1,'configuration':cfg})
                m.eval()
                for name,p in m.named_parameters():p.requires_grad_(name.startswith('head.'))
                opt=optimizer(m,head_only=True);stage=2
            kind='head' if stage==2 else kind_for(recipe)
            stage_n=cfg['N_head'] if stage==2 else cfg['N_FT'] if recipe=='B5' else n
            local=u-cfg['N_FT']+1 if stage==2 else u+1
            tick=time.perf_counter();out=update(m,opt,data,seed,u,kind,cache,math.ceil(.05*n),local,stage_n)
            loop_seconds+=time.perf_counter()-tick
            trace.append({'update':u+1,'stage':stage,**out})
            if u+1 in points:
                seq+=1;checkpoint(folder,seq,m,opt,seed,recipe,u+1,stage,stage_n,cache,loop_seconds,began,lock)
                dump(folder/'training-trace.json',trace)
            if u+1==lock['J']:
                save_pt(folder/'J.pt',{'model':m.state_dict(),'completed_updates':u+1,'stage':stage,'initial_state_sha256':init})
                jp=predict(m,data,data.splits['validation'],folder/'J-validation.pt');jresult={k:jp[k] for k in ['correct','count','seconds']};jresult['stage']=stage
            if time.monotonic()-last_progress>=30:
                status('formal_training',seed=seed,recipe=recipe,completed_updates=u+1,N=n,formal_runs_completed=len(completed),mps_used_seconds=used_seconds());last_progress=time.monotonic()
        final_hash=save_pt(folder/'final.pt',{'model':m.state_dict(),'completed_updates':n,'stage':stage,'initial_state_sha256':init})
        pred=predict(m,data,data.splits['validation'],folder/'final-validation.pt')
        record.update(completed_updates=n,status='complete',loop_seconds=loop_seconds,full_run_seconds=used_seconds()-record['mps_started_seconds'],
                      full_run_wall_seconds=time.time()-datetime.fromisoformat(record['started_utc']).timestamp(),resumed=cp is not None,
                      final_file_sha256=final_hash,final_state_sha256=tensorhash(m.state_dict()),final_validation={k:pred[k] for k in ['correct','count','seconds']},J_validation=jresult)
        dump(folder/'complete.json',record);completed.append(record)
        # Permanent final/J weights, stage boundary, traces and predictions are retained.
        for p in folder.glob('recovery-*.pt*'):p.unlink()
        dump(folder/'active-recovery-closed.json',{'final_and_J_verified':True,'removed_only_completed_active_recovery_slots':True})
        dump(ROOT/'formal-progress.json',{'completed':len(completed),'runs':completed})
        status('formal_run_complete',seed=seed,recipe=recipe,formal_runs_completed=len(completed),mps_used_seconds=used_seconds())
        del m,opt,pred;gc.collect();torch.mps.empty_cache()
    select_and_test(lock,data,completed)

def select_and_test(lock,data,completed):
    assert len(completed)==24
    totals={r:sum(x['final_validation']['correct'] for x in completed if x['recipe']==r) for r in RECIPES}
    baseline=min(RECIPES[:6],key=lambda r:(-totals[r],r));candidate=min(RECIPES[6:],key=lambda r:(-totals[r],r))
    selection={'locked_utc':datetime.now(timezone.utc).isoformat(),'baseline':baseline,'candidate':candidate,'val_correct_sum':totals,
               'val_denominator':3*len(data.splits['validation']),'tie_rule':'integer correct then lexicographic ID',
               'final_model_sha256':{f'{x["seed"]}-{x["recipe"]}':x['final_file_sha256'] for x in completed},'protocol_lock_sha256':sha(ROOT/'protocol.lock.json')}
    if (ROOT/'selection.json').exists():
        old=json.loads((ROOT/'selection.json').read_text())
        for k in ['baseline','candidate','val_correct_sum','final_model_sha256','protocol_lock_sha256']:assert old[k]==selection[k]
        selection=old;selhash=sha(ROOT/'selection.json');assert json.loads((ROOT/'selection.sha256.json').read_text())['sha256']==selhash
    else:
        dump(ROOT/'selection.json',selection);selhash=sha(ROOT/'selection.json');dump(ROOT/'selection.sha256.json',{'sha256':selhash})
    status('selection_locked_before_test',selection_sha256=selhash,baseline=baseline,candidate=candidate,formal_runs_completed=24)
    tests=[]
    for seed in [11,22,33]:
        for recipe in [baseline,candidate]:
            assert sha(ROOT/'selection.json')==selhash
            p=run_path(seed,recipe)/'final.pt';assert sha(p)==selection['final_model_sha256'][f'{seed}-{recipe}']
            tp=run_path(seed,recipe)/'test.pt'
            if tp.exists():
                assert sha(tp)==json.loads(tp.with_suffix('.pt.sha256.json').read_text())['sha256']
                pr=torch.load(tp,map_location='cpu',weights_only=True);assert pr['ids']==data.splits['test'];duration=pr['seconds']
            else:
                m=model(seed,p);began=time.perf_counter();pr=predict(m,data,data.splits['test'],tp);duration=time.perf_counter()-began;del m
            tests.append({'seed':seed,'recipe':recipe,'correct':pr['correct'],'count':pr['count'],'seconds':duration,'prediction_sha256':sha(tp)})
            dump(ROOT/'test-progress.json',tests);gc.collect();torch.mps.empty_cache()
    paired=[]
    for seed in [11,22,33]:
        b=next(x for x in tests if x['seed']==seed and x['recipe']==baseline);c=next(x for x in tests if x['seed']==seed and x['recipe']==candidate)
        paired.append({'seed':seed,'baseline_correct':b['correct'],'candidate_correct':c['correct'],'count':b['count'],'delta_pp':100*(c['correct']-b['correct'])/b['count']})
    mean=float(np.mean([x['delta_pp'] for x in paired]));passed=mean>=1 and all(x['delta_pp']>0 for x in paired)
    result={'status':'complete','baseline':baseline,'candidate':candidate,'paired_test':paired,'mean_delta_pp':mean,'usefulness_gate_passed':passed,
            'scientific_status':'practical_followup_gate_passed' if passed else 'practical_followup_gate_not_met','significance_claim':False,
            'originality_established':False,'selection_sha256':selhash,'test_models':tests,'actual_runs':completed,'mps_used_seconds_at_report':used_seconds()}
    dump(ROOT/'results.json',result)
    status('complete',formal_runs_completed=24,test_metrics_computed=True,scientific_status=result['scientific_status'],mean_test_delta_pp=mean)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','formal']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    active=json.loads((BASE/'mps-ledger.json').read_text()).get('active')
    assert active and active.get('pid')==os.getpid(),'Must run through supervise.py; direct unaccounted execution is forbidden'
    try:
        if args.stage=='preflight':preflight()
        else:formal(resume=args.resume)
    except Exception as exc:
        dump(ROOT/f'{args.stage}-failure.json',{'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'mps_used_seconds':used_seconds(),
             'scientific_status':'undetermined','no_cpu_model_fallback':True,'no_protocol_relaxation':True})
        status('blocked_'+args.stage,scientific_status='undetermined',failure=str(exc));raise
