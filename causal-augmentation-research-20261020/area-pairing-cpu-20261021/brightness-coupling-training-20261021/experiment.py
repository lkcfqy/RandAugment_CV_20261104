"""One fixed brightness-coupling pilot; no adaptive recipe/test selection."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0')
from pathlib import Path
import collections,datetime,gc,hashlib,importlib.util,json,re,resource,signal,subprocess,sys,time,traceback
import numpy as np
from PIL import Image,ImageDraw
import torch,torchvision
from torchvision.transforms import functional as F,InterpolationMode

ROOT=Path(__file__).resolve().parent
BASE=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
DATA=BASE/'mechanism-transfer-pets-20261021'
HELPER=BASE/'bn-affine-training-diagnostic-20261021/diagnostic.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='7e2eaf44e41c887eee545541d68d4d7bc8f1276cc601231c1b777243de336c5d'
spec=importlib.util.spec_from_file_location('fixed_training_helpers',HELPER);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
ARMS=['shared','semantic_independent','shifted_independent']
START=None;MODELS={};OPTS={};STEPS={};CHECKPOINTS={};RESOURCE=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgbsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,x):
    with (ROOT/name).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def load(name):return json.loads((ROOT/name).read_text())
def seed(key):return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
def configure():
    try:os.nice(10)
    except OSError:pass
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
def shift(a):return np.roll(a,(112,112),(0,1))

def masks(label):
    assert label.shape==(224,224) and set(np.unique(label)).issubset({1,2,3})
    # Deterministic nearest known trimap class under 8-neighbor/Chebyshev
    # distance; unknown ties go to foreground. This estimates the uncertain
    # boundary, rather than occluding the entire broad trimap unknown band.
    resolved=np.where(label==3,0,label).astype(np.uint8)
    assert np.any(resolved!=0)
    while np.any(resolved==0):
        padded=np.pad(resolved,1,constant_values=0);hf=np.zeros_like(resolved,dtype=bool);hb=hf.copy()
        for dy in range(3):
            for dx in range(3):
                nb=padded[dy:dy+224,dx:dx+224];hf|=nb==1;hb|=nb==2
        unknown=resolved==0;assert np.any(unknown&(hf|hb))
        resolved[unknown&hf]=1;resolved[unknown&~hf&hb]=2
    g=np.zeros_like(label,dtype=bool)
    for dy in range(-2,3):
        for dx in range(-2,3):g|=resolved!=np.roll(resolved,(dy,dx),(0,1))
    g=g|shift(g);assert np.array_equal(g,shift(g))
    f=(resolved==1)&~g;b=(resolved==2)&~g;sf=shift(f);sb=shift(b)
    assert np.all(f.astype(int)+b+g==1) and np.all(sf.astype(int)+sb+g==1)
    assert f.sum()==sf.sum() and b.sum()==sb.sum()
    return f,b,sf,sb,g

def read_pair(photo,mask):
    with Image.open(photo) as im:
        origsize=im.size;im=im.convert('RGB');im=F.center_crop(F.resize(im,256,InterpolationMode.BILINEAR,antialias=True),[224,224]);a=np.array(im)
    with Image.open(mask) as im:
        assert im.size==origsize
        label=np.array(F.center_crop(F.resize(im,256,InterpolationMode.NEAREST),[224,224]))
    return a,label,masks(label)

def brighten(a,k):return np.minimum((a.astype(np.uint16)*[3,5,7][k]+2)//5,255).astype(np.uint8)
def render(a,ms,arm,j,k):
    f,b,sf,sb,g=ms;fa,ba=(sf,sb) if arm=='shifted_independent' else (f,b)
    out=np.full_like(a,128);out[fa]=brighten(a,j)[fa];out[ba]=brighten(a,k)[ba]
    return out
def tensor(a):return D.tensor(Image.fromarray(a))

def schedules(rows):
    # 28 independently shuffled complete factorial blocks per original.
    out=np.empty((252,64,2),np.int8)
    for i,r in enumerate(rows):
        for cycle in range(28):
            rng=np.random.default_rng(seed('brightness-coupling-schedule-v1:'+r['source_id']+':'+str(cycle)))
            j=rng.permutation(np.repeat(np.arange(3),3));k=np.empty(9,np.int8)
            for v in range(3):k[np.flatnonzero(j==v)]=rng.permutation(3)
            out[cycle*9:(cycle+1)*9,i,0]=j;out[cycle*9:(cycle+1)*9,i,1]=k
    rng=np.random.default_rng(2026102102);orders=np.array([rng.permutation(64) for _ in range(252)],np.int64)
    return out,orders

def make_protocol():
    rows=json.loads((BASE/'stage-validity-pilot-20261021/protocol.json').read_text())['train_rows']
    index=json.loads((DATA/'zip-index-zip64.json').read_text());e=index['entries']
    used={p.stem for p in (DATA/'photos').glob('*.jpg')}|{p.stem for p in (DATA/'masks').glob('*.png')}|{p.stem for p in (BASE/'train-filter-sourceweight-20261021/test-photos').glob('*.jpg')}
    crcs={e['images/'+s+'.jpg']['crc32'] for s in used if 'images/'+s+'.jpg' in e};pool=[]
    for line in (DATA/'archive-text-2.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        v=line.split();sid=v[0];im='images/'+sid+'.jpg';ma='annotations/trimaps/'+sid+'.png'
        if sid in used or im not in e or ma not in e or e[im]['crc32'] in crcs:continue
        if max(e[im]['bytes'],e[ma]['bytes'])>512000:continue
        pool.append(dict(source_id=sid,Y=int(v[2])-1,breed=int(v[1]),image_entry=im,mask_entry=ma,
            rank=hashlib.sha256(('brightness-coupling-heldout-v1:'+sid).encode()).hexdigest()))
    test=[];seen=set()
    for y in [0,1]:
        count=0
        for r in sorted((r for r in pool if r['Y']==y),key=lambda r:r['rank']):
            crc=e[r['image_entry']]['crc32']
            if crc in seen:continue
            seen.add(crc);r['image_metadata']=e[r['image_entry']];r['mask_metadata']=e[r['mask_entry']];test.append(r);count+=1
            if count==24:break
        assert count==24
    http=sum(30+len(r[k+'_entry'])+512+r[k+'_metadata']['compressed_bytes'] for r in test for k in ['image','mask'])
    assert http<8_000_000
    p=dict(at_UTC=now(),question='At exactly matched per-original pixelwise brightness marginals, does independent semantic-region brightness reduce trained background sensitivity more than shared brightness AND the same geometry shifted away from the semantic boundary?',
        status='controlled replication and candidate calibration; not a novel augmentation method',
        arms=ARMS,train_rows=rows,test_rows=test,excluded_historical_photo_or_mask_IDs=sorted(used),
        test_selection='24/species from official test, fixed SHA256 key, no historical photo/mask ID or JPEG CRC; metadata-only before lock; one evaluation',
        factors=[.6,1.,1.4],pixel_formula='uint8 min(255, floor((RGB*[3,5,7]+2)/5)); nearest integer; no interpolation after brightness',
        preprocess='RGB shorter side256 bilinear center224; trimap same resize geometry nearest; ImageNet normalization after compositing',
        partitions='First resolve trimap3 unknown by simultaneous nearest known1/2 wavefront in8-neighbor/Chebyshev distance, without wrapping; ties foreground. This estimates an uncertain boundary, not new groundtruth. Shift this binary partition periodically by(112,112). Fixed common neutral128 guard is union of radius2 Chebyshev binary-boundary neighborhood and its translation. Guard invariant under involution, so eligible region areas exactly match. Periodic guard treatment is artificial and common to all arms.',
        shared_arm_reuse='Shared output is bit-identical for semantic and shifted partitions; train once and reuse, not an independent replicate.',
        schedule='Per original each9-pass block: foreground factor sequence random permutation of eachfactor3times; shared bg=fg; independent bg permutes all3 within eachfggroup, yielding all9pairs once. Identical schedules for semantic/shifted, identical source order allarms. 28 complete blocks=252passes=2016updates, eachfactor84times/source/region/pixel. RNG independent of label; class-conditional empirical parameter marginals also identical.',
        training=dict(initialization='existing pretrained ResNet18 V1 + headseed0; all3sameinitialstate',seed_count=1,batch=8,updates=2016,passes=252,head_LR=.01,body_LR=.0001,
            warmup='first128updates factor .1+.9*(step-1)/127 then1',SGD_momentum=.9,weight_decay=.0001,BN='runningstatsfixed,affinetrainable',precision='FP32',device='MPS',checkpoints=[144,1008,2016],
            effective_presentations_per_arm=16128,selection='fixed final2016; no bestcheckpoint/test tuning; no additional seed or recipe'),
        training_validity='allstepsloss/grad/paramsfinite; final guarded-clean and fixed9-view train probe LL <=.5initial; final512online mean<=.5first256; cleanlogitsnonconstant; meanfeaturevar>1e-4initial. These are task-specific feasibility checks, not universal accuracy thresholds.',
        background_bank='4fixed64x64 all-label2 patches from existing training photos,2per source species, sorted by SHA256 coupling-background-v1:ID; first allbackground window on8px grid; resize224 bilinear. No outcome selection. Fixed4texture donors, not a sample of natural environments; source label not a scene label.',
        intervention='For each heldout original use unchanged semantic-F pixels, same common guard128, replace semantic-B pixels with each of4fixed real background patches. No foreground rescale/reposition. Allarms evaluated on identical composites; confirm foreground/guardbyte equality.',
        primary_metric='Per original SD(population,4donors) of logit margin z1-z0, divided by model mean absolute margin on its64 common-guarded clean training originals. Fixed train-only positive scale cancels a global logit rescaling. Report raw SD and all logits as checks.',
        contrasts='A=mean(S_shared-S_semantic); B=mean(S_shifted-S_semantic). Positive A suggests benefit of independent semantic coupling; positive B is extra benefit beyond this shifted-partition control. No pure semantic mechanism identification: geometry/relative contrast/joint diversity are treatment components.',
        secondary_metrics='original photo LL/accuracy; guarded-original LL/accuracy; neutral-background foreground-only accuracy; mean intervention accuracy; per-source any donor label-flip; mean max-min p(cat) range; raw marginSD. Report every arm, no metric switching.',
        uncertainty='5000paired species-stratified source-photo bootstraps seed2026102117, percentile95 intervals for A andB. 48originals unit, not192composites. Conditional on one training seed and4fixed donor backgrounds; no seed/background-population uncertainty.',
        candidate_screen=dict(min_relative_primary_reduction=.20,both_A_B_bootstrap95_lower_positive=True,
            semantic_original_and_foregroundonly_accuracy_drop_vs_shared_max=.05,
            measurement_floor_shared_normalized_SD=.001,
            reason='predefined large-effect small-pilot screen, not a power guarantee or proof of originality; if unmet close this narrow pilot without retuning'),
        final_access_order='prepare/train-only checks -> protocol and all executable hashes lock -> train all3 and verify gates -> save training-freeze hashes -> bounded fetch fresh48JPEG+48masks -> fixed single evaluation. Failed training or implementation prevents final access.',
        data=dict(official='https://www.robots.ox.ac.uk/~vgg/data/pets/',license='CC BY-SA4.0; Parkhi,Vedaldi,Zisserman,Jawahar; originals retain copyright',
            mirror='antokun/The-Oxford-IIIT-Pet-Dataset-With-Annotations',revision=index['revision'],archive_bytes_not_to_download=815141550,
            new_HTTP_range_bytes=http,JPEG_payload_bytes=sum(r['image_metadata']['bytes'] for r in test),mask_payload_bytes=sum(r['mask_metadata']['bytes'] for r in test)),
        resources=dict(CPU_threads=1,interop_threads=1,nice_increment=10,workers=0,training_hard_wall_seconds=600,soft_stop_wall_seconds=540,
            memory_free_stop_percent=15,A100=False,ARIS=False,software_changes=False,other_project_access=False),
        difference_from_closed_work='Old area study tested source/scale associations and frozen-model effects; Equalize study crossed already-processed regions at inference with a frozen classifier and was negative. Here pixels undergo only pointwise Brightness during end-to-end training, followed by a separately frozen background intervention evaluation. No old negative claim revived.',
        closest_prior=dict(YOCO='https://arxiv.org/html/2201.12078v3',SLAug='https://arxiv.org/html/2211.14805v1',SSH='https://arxiv.org/html/2108.06805v2',
            boundary='YOCO already uses the same marginal augmentation law independently per piece and tests background replacement. Exact finite balance is a control, not method novelty. Remaining pilot isolates brightness-only semantic alignment versus matched shifted geometry; not established original contribution.'),
        model_request='GPT-6-Astra/max; runtime execution receipt unavailable')
    save('protocol.json',p)
    print(json.dumps(dict(status='protocol_saved_metadata_only',fresh_photos=len(test),new_HTTP_range_bytes=http,protocol_sha256=sha(ROOT/'protocol.json'))),flush=True)

def training_data():
    p=load('protocol.json');images=[];labels=[];maps=[];receipts=[]
    for r in p['train_rows']:
        ph=DATA/'photos'/(r['source_id']+'.jpg');ma=DATA/'masks'/(r['source_id']+'.png')
        a,l,m=read_pair(ph,ma);images.append(a);labels.append(l);maps.append(m)
        receipts.append(dict(source_id=r['source_id'],Y=r['Y'],photo_sha256=sha(ph),mask_sha256=sha(ma),RGB_sha256=rgbsha(a)))
    return p,images,labels,maps,receipts

def prepare():
    configure();make_protocol();p,ims,labs,maps,receipts=training_data();sc,orders=schedules(p['train_rows'])
    factors=collections.Counter();marginal_checks=0;joint_checks=0;geo=[];hashes=[]
    for i,(r,a,ms) in enumerate(zip(p['train_rows'],ims,maps)):
        f,b,sf,sb,g=ms
        assert f.any() and b.any()
        geo.append(dict(source_id=r['source_id'],FG_pixels=int(f.sum()),BG_pixels=int(b.sum()),guard_pixels=int(g.sum()),
            semantic_shifted_FG_IoU=float((f&sf).sum()/(f|sf).sum())))
        for cycle in range(28):
            pairs=sc[cycle*9:(cycle+1)*9,i]
            assert collections.Counter(map(tuple,pairs.tolist()))==collections.Counter({(j,k):1 for j in range(3) for k in range(3)})
            assert all(np.bincount(pairs[:,v],minlength=3).tolist()==[3,3,3] for v in [0,1]);joint_checks+=1
        # Byte hashes of entire region populations, not just magnitude counts.
        for region,sel in [('F',f),('B',b)]:
            ss=[];ii=[]
            for j,k in sc[:9,i]:
                ss.append(rgbsha(render(a,ms,'shared',int(j),int(j))[sel]))
                ii.append(rgbsha(render(a,ms,'semantic_independent',int(j),int(k))[sel]))
            assert sorted(ss)==sorted(ii);marginal_checks+=1
        for j in range(3):
            assert np.array_equal(render(a,ms,'shared',j,j),render(a,ms,'shifted_independent',j,j))
        # Every eligible pixel gets each factor3/cycle in every arm, because each region does.
        hashes.append(dict(source_id=r['source_id'],partition_masks_sha256=rgbsha(np.stack(ms)),
            common_guarded_clean_sha256=rgbsha(render(a,ms,'shared',1,1))))
    donors=[]
    for y in [0,1]:
        count=0
        for i in sorted((i for i,r in enumerate(p['train_rows']) if r['Y']==y),key=lambda i:hashlib.sha256(('coupling-background-v1:'+p['train_rows'][i]['source_id']).encode()).hexdigest()):
            found=None
            for top in range(0,161,8):
                for left in range(0,161,8):
                    if np.all(labs[i][top:top+64,left:left+64]==2):found=(top,left);break
                if found is not None:break
            if found is None:continue
            top,left=found;patch=ims[i][top:top+64,left:left+64]
            bg=np.array(Image.fromarray(patch).resize((224,224),Image.Resampling.BILINEAR))
            path=ROOT/('donor'+str(len(donors))+'.png');Image.fromarray(bg).save(path)
            donors.append(dict(source_id=p['train_rows'][i]['source_id'],source_species=y,top=top,left=left,side=64,
                all_original_mask_label2=True,file=path.name,sha256=sha(path),RGB_sha256=rgbsha(bg)))
            count+=1
            if count==2:break
        assert count==2,'Insufficient prespecified background patches; stop, no patch-size tuning.'
    save('donors.json',donors)
    audit=dict(status='all_training_side_implementation_checks_passed',at_UTC=now(),source_photos=receipts,geometry=geo,pixel_hashes=hashes,
        complete_factorial_checks=joint_checks,FG_BG_exact_processed_byte_marginal_checks=marginal_checks,
        shared_two_partition_outputs_identical=True,every_eligible_pixel_factor_counts_per_arm=[84,84,84],
        Y_parameter_marginal_independent_by_exact_class_counts=True,conditions_do_not_create_claimed_Y_background_source_marginal_association=True,
        parameters_are_design_variables_not_hidden_test_features=True,schedule_sha256=rgbsha(sc),orders_sha256=rgbsha(orders),
        final_pixels_read=0,fresh_test_downloaded=False,helper_sha256=sha(HELPER))
    save('implementation-checks.json',audit)
    # Fixed first cat/dog previews are training data only. No metric-based selection.
    picks=[0,32];sheet=Image.new('RGB',(224*5,250*2),'white');draw=ImageDraw.Draw(sheet)
    for q,i in enumerate(picks):
        views=[ims[i],render(ims[i],maps[i],'shared',0,0),render(ims[i],maps[i],'semantic_independent',0,2),render(ims[i],maps[i],'shifted_independent',0,2)]
        f,b,sf,sb,g=maps[i];bg=np.array(Image.open(ROOT/donors[0]['file']).convert('RGB'));v=np.full_like(ims[i],128);v[f]=ims[i][f];v[b]=bg[b];views.append(v)
        for col,(name,v) in enumerate(zip(['original','shared .6/.6','semantic .6/1.4','shifted .6/1.4','background swap'],views)):
            sheet.paste(Image.fromarray(v),(col*224,q*250+26));draw.text((col*224+4,q*250+4),name,fill='black')
    sheet.save(ROOT/'training-implementation-preview.png')
    print(json.dumps(dict(status=audit['status'],complete_factorial_checks=joint_checks,byte_marginal_checks=marginal_checks,
        mean_guard_fraction=float(np.mean([x['guard_pixels']/50176 for x in geo])),mean_shifted_FG_IoU=float(np.mean([x['semantic_shifted_FG_IoU'] for x in geo])),donors=[r['source_id'] for r in donors])),flush=True)

def freeze():
    a=load('implementation-checks.json');assert a['status']=='all_training_side_implementation_checks_passed'
    hashes={n:sha(ROOT/n) for n in ['protocol.json','experiment.py','implementation-checks.json','donors.json','prior-overlap.md','development-visual-check.json']}
    save('protocol-lock.json',dict(at_UTC=now(),hashes=hashes,final_pixels_read=0,recipe_and_metrics_fixed=True))
    print(json.dumps(dict(status='protocol_locked_before_training_and_test',sha256=sha(ROOT/'protocol-lock.json'))),flush=True)
def checklock():
    for n,h in load('protocol-lock.json')['hashes'].items():assert sha(ROOT/n)==h,n

def resource_check(step):
    r=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True,timeout=3)
    m=re.search(r'System-wide memory free percentage:\s*(\d+)%',r.stdout);free=int(m.group(1)) if m and r.returncode==0 else None
    RESOURCE.append(dict(step_per_arm=step,memory_free_percent=free,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if free is not None and free<15:raise RuntimeError('memory_free_below15percent;yield_to8A')
def guard():
    if time.perf_counter()-START>540:raise TimeoutError('fixed540second_soft_wall;no_budget_extension')
def timeout(signum,frame):raise TimeoutError('fixed600second_hard_wall;no_budget_extension')
def checkpoint(arm,step,final=False):
    state={n:v.detach().cpu().clone() for n,v in MODELS[arm].state_dict().items()}
    x=dict(arm=arm,step=step,model=state,model_state_sha256=D.state_hash(state),protocol_sha256=sha(ROOT/'protocol.json'),
        code_sha256=sha(Path(__file__)),BN_sha256=D.bn_stats(MODELS[arm]))
    if final:
        raw=OPTS[arm].state_dict();x['optimizer']={'param_groups':raw['param_groups'],'state':{}}
        for k,v in raw['state'].items():x['optimizer']['state'][k]={n:a.detach().cpu().clone() if torch.is_tensor(a) else a for n,a in v.items()}
    name=arm+'-step'+str(step)+'.pt';assert not (ROOT/name).exists();torch.save(x,ROOT/name)
    CHECKPOINTS.setdefault(arm,[]).append(dict(file=name,bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name),state_sha256=x['model_state_sha256'],optimizer_saved=final))

def predict(model,x,features=False):
    model.eval();zs=[];fs=[];handle=None
    if features:handle=model.avgpool.register_forward_hook(lambda m,args,out:fs.append(out.detach().flatten(1).cpu().numpy().copy()))
    with torch.inference_mode():
        for k in range(0,len(x),8):zs.append(model(x[k:k+8].to('mps')).cpu())
    if handle:handle.remove()
    D.train_mode(model);z=torch.cat(zs).numpy();assert np.isfinite(z).all()
    return z,np.concatenate(fs) if features else None
def metrics(z,y):
    loss=torch.nn.functional.cross_entropy(torch.from_numpy(z),torch.tensor(y,dtype=torch.long),reduction='none').numpy()
    return dict(mean_LL_nats=float(loss.mean()),accuracy=float((z.argmax(1)==y).mean()))
def train():
    global START
    configure();checklock();START=time.perf_counter();signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    assert torch.backends.mps.is_available();resource_check(0)
    p,ims,labs,maps,receipts=training_data();assert receipts==load('implementation-checks.json')['source_photos']
    sc,orders=schedules(p['train_rows']);assert rgbsha(sc)==load('implementation-checks.json')['schedule_sha256']
    ys=np.array([r['Y'] for r in p['train_rows']],np.int64);y=torch.tensor(ys)
    clean=torch.stack([tensor(render(a,m,'shared',1,1)) for a,m in zip(ims,maps)])
    # Small uint8 bank; no worker processes or GPU preloading of full dataset.
    banks={n:np.stack([[render(a,m,n,j,k) for j in range(3) for k in range(3)] for a,m in zip(ims,maps)]) for n in ['semantic_independent','shifted_independent']}
    probe={}
    for arm in ARMS:
        bb=banks['semantic_independent'] if arm=='shared' else banks[arm]
        probe[arm]=torch.stack([tensor(bb[i,int(j)*3+int(j if arm=='shared' else k)]) for i in range(64) for j,k in sc[:9,i]])
    py=np.repeat(ys,9);results={};arrays={};losses={n:[] for n in ARMS};compute={n:0. for n in ARMS};chains={n:hashlib.sha256() for n in ARMS}
    handles={n:(ROOT/(n+'-steps.jsonl')).open('x') for n in ARMS}
    for arm in ARMS:
        m,opt,initial=D.make_model(False);assert initial=='ec9897fa3d388d789d326977f28f7814ed3b474c083763c2468e46744999fe32'
        MODELS[arm]=m;OPTS[arm]=opt;STEPS[arm]=0
        z,ff=predict(m,clean,True);zp,_=predict(m,probe[arm]);arrays[arm+'_0_clean']=z;arrays[arm+'_0_probe']=zp
        results[arm]=dict(initial_sha256=initial,BN_initial_sha256=D.bn_stats(m),initial_feature_variance=float(ff.astype(float).var(0).mean()),
            stages={'0':dict(clean=metrics(z,ys),own_probe=metrics(zp,py))})
    for block in range(14):
        for arm in ARMS:
            bb=banks['semantic_independent'] if arm=='shared' else banks[arm]
            for epoch in range(block*18,(block+1)*18):
                for off in range(0,64,8):
                    guard();ids=orders[epoch,off:off+8];views=[]
                    for i in ids:
                        j,k=map(int,sc[epoch,i]);k=j if arm=='shared' else k;v=bb[i,j*3+k];views.append(v)
                        chains[arm].update((str(epoch)+':'+p['train_rows'][i]['source_id']).encode());chains[arm].update(v.tobytes())
                    x=torch.stack([tensor(v) for v in views]);s=STEPS[arm]+1;fac=.1+.9*(s-1)/127 if s<=128 else 1.
                    opt=OPTS[arm];opt.param_groups[0]['lr']=.01*fac;opt.param_groups[1]['lr']=.0001*fac
                    torch.mps.synchronize();t=time.perf_counter();ll=D.step(MODELS[arm],opt,x,y[ids]);torch.mps.synchronize();dt=time.perf_counter()-t
                    compute[arm]+=dt;STEPS[arm]=s;losses[arm].append(ll)
                    handles[arm].write(json.dumps(dict(step=s,epoch=epoch,loss=ll,head_lr=.01*fac,body_lr=.0001*fac,compute_seconds=dt))+'\n')
            handles[arm].flush();s=STEPS[arm]
            if s in [144,1008,2016]:
                z,ff=predict(MODELS[arm],clean,True);zp,_=predict(MODELS[arm],probe[arm]);arrays[arm+'_'+str(s)+'_clean']=z;arrays[arm+'_'+str(s)+'_probe']=zp
                results[arm]['stages'][str(s)]=dict(clean=metrics(z,ys),own_probe=metrics(zp,py),
                    feature_variance=float(ff.astype(float).var(0).mean()),unique_logit_rows=len(np.unique(z,axis=0)),mean_abs_train_margin=float(np.abs(z[:,1]-z[:,0]).mean()))
                checkpoint(arm,s,final=s==2016)
        s=(block+1)*144;assert all(STEPS[n]==s for n in ARMS)
        print(json.dumps(dict(step_per_arm=s,online_last144_LL={n:float(np.mean(losses[n][-144:])) for n in ARMS},wall_seconds=time.perf_counter()-START)),flush=True)
        if s in [576,1152,1728,2016]:resource_check(s)
    gates={}
    for arm in ARMS:
        a=results[arm];f=a['stages']['2016'];ini=a['stages']['0'];assert D.bn_stats(MODELS[arm])==a['BN_initial_sha256']
        gates[arm]=dict(all2016updates_finite=True,clean_learned=f['clean']['mean_LL_nats']<=.5*ini['clean']['mean_LL_nats'],own_probe_learned=f['own_probe']['mean_LL_nats']<=.5*ini['own_probe']['mean_LL_nats'],
            online_learned=float(np.mean(losses[arm][-512:]))<=.5*float(np.mean(losses[arm][:256])),no_collapse=f['feature_variance']>1e-4*a['initial_feature_variance'] and f['unique_logit_rows']>1)
        a.update(criteria=gates[arm],normalizer=f['mean_abs_train_margin'],training_compute_seconds=compute[arm],
            first256_online_LL=float(np.mean(losses[arm][:256])),last512_online_LL=float(np.mean(losses[arm][-512:])),
            checkpoints=CHECKPOINTS[arm],training_RGB_source_chain_sha256=chains[arm].hexdigest(),BN_stats_unchanged=True)
        handles[arm].close()
    np.savez_compressed(ROOT/'training-predictions.npz',**arrays,clean_labels=ys,probe_labels=py)
    out=dict(status='completed_fixed_training',at_UTC=now(),all_training_gates_passed=all(all(g.values()) for g in gates.values()),arms=results,
        updates_per_arm=STEPS,main_wall_seconds=time.perf_counter()-START,guarded_training_compute_seconds=sum(compute.values()),
        process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource_checks=RESOURCE,torch=torch.__version__,torchvision=torchvision.__version__,
        device='MPS',CPU_threads=1,worker_processes=0,final_pixels_read=0,protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)))
    save('training-results.json',out);signal.alarm(0)
    if out['all_training_gates_passed']:
        save('training-freeze.json',dict(at_UTC=now(),protocol_lock_sha256=sha(ROOT/'protocol-lock.json'),training_results_sha256=sha(ROOT/'training-results.json'),
            prediction_sha256=sha(ROOT/'training-predictions.npz'),final_models={n:CHECKPOINTS[n][-1] for n in ARMS},final_pixels_read=0))
    print(json.dumps(dict(status=out['status'],all_training_gates_passed=out['all_training_gates_passed'],wall_seconds=out['main_wall_seconds'],compute_seconds=out['guarded_training_compute_seconds'],
        final_train={n:results[n]['stages']['2016'] for n in ARMS})),flush=True)

def fetch():
    checklock();freeze=load('training-freeze.json');assert freeze['protocol_lock_sha256']==sha(ROOT/'protocol-lock.json') and freeze['training_results_sha256']==sha(ROOT/'training-results.json')
    for v in freeze['final_models'].values():assert sha(ROOT/v['file'])==v['sha256']
    spec=importlib.util.spec_from_file_location('bounded_zip',DATA/'remote_zip.py');Z=importlib.util.module_from_spec(spec);spec.loader.exec_module(Z)
    p=load('protocol.json');files=[];(ROOT/'heldout-photos').mkdir();(ROOT/'heldout-masks').mkdir()
    for num,row in enumerate(p['test_rows']):
        for kind,sub,ext in [('image','heldout-photos','.jpg'),('mask','heldout-masks','.png')]:
            z=Z.RemoteZip(p['data']['revision'],{row[kind+'_entry']:row[kind+'_metadata']})
            payload=z.member_fast(row[kind+'_entry'],512000);path=ROOT/sub/(row['source_id']+ext)
            with path.open('xb') as f:f.write(payload)
            files.append(dict(source_id=row['source_id'],kind=kind,bytes=len(payload),sha256=sha(path),CRC_verified=True,range_requests=z.receipts))
        if (num+1)%12==0:print(json.dumps(dict(fetched_photo_mask_pairs=num+1,total=48)),flush=True)
    total=sum(q['bytes'] for x in files for q in x['range_requests']);assert total==p['data']['new_HTTP_range_bytes']
    save('heldout-fetch-receipt.json',dict(status='all96files_CRC_SHA_verified',at_UTC=now(),files=files,HTTP_range_bytes=total,archive_downloaded=False,
        training_freeze_sha256=sha(ROOT/'training-freeze.json'),official_license='CC BY-SA4.0',official_source=p['data']['official']))
    print(json.dumps(dict(status='fetch_completed',HTTP_range_bytes=total,files=len(files))),flush=True)

def evaluate_final():
    configure();checklock();tr=load('training-results.json');assert tr['all_training_gates_passed'];p=load('protocol.json');fr=load('heldout-fetch-receipt.json')
    assert fr['training_freeze_sha256']==sha(ROOT/'training-freeze.json')
    photos={r['source_id']:r['sha256'] for r in fr['files'] if r['kind']=='image'};masks_h={r['source_id']:r['sha256'] for r in fr['files'] if r['kind']=='mask'}
    donor=[]
    for r in load('donors.json'):
        assert sha(ROOT/r['file'])==r['sha256'];donor.append(np.array(Image.open(ROOT/r['file']).convert('RGB')))
    originals=[];guarded=[];fgonly=[];composites=[];audit=[];ys=[]
    for r in p['test_rows']:
        sid=r['source_id'];ph=ROOT/'heldout-photos'/(sid+'.jpg');ma=ROOT/'heldout-masks'/(sid+'.png')
        assert sha(ph)==photos[sid] and sha(ma)==masks_h[sid]
        a,l,ms=read_pair(ph,ma);f,b,sf,sb,g=ms;ys.append(r['Y']);originals.append(tensor(a));guarded.append(tensor(render(a,ms,'shared',1,1)))
        neutral=np.full_like(a,128);neutral[f]=a[f];fgonly.append(tensor(neutral));pair=[]
        for bg in donor:
            v=neutral.copy();v[b]=bg[b];assert np.array_equal(v[f],a[f]) and np.all(v[g]==128);composites.append(tensor(v));pair.append(rgbsha(v))
        audit.append(dict(source_id=sid,Y=r['Y'],FG_pixels=int(f.sum()),BG_pixels=int(b.sum()),guard_pixels=int(g.sum()),
            FG_fraction=float(f.mean()),FG_pixels_byte_identical_all4=True,guard_byte_identical_all4=True,composite_RGB_sha256=pair))
    ys=np.array(ys);xx={'original':torch.stack(originals),'guarded':torch.stack(guarded),'foreground_only':torch.stack(fgonly),'background_swap':torch.stack(composites)}
    arrays={'labels':ys};out={}
    for arm in ARMS:
        ck=load('training-freeze.json')['final_models'][arm];assert sha(ROOT/ck['file'])==ck['sha256']
        saved=torch.load(ROOT/ck['file'],map_location='cpu',weights_only=True);m=torchvision.models.resnet18(weights=None);m.fc=torch.nn.Linear(512,2);m.load_state_dict(saved['model'],strict=True)
        assert D.state_hash(m.state_dict())==ck['state_sha256'];m.to('mps');armout={}
        for kind,x in xx.items():
            z,_=predict(m,x);arrays[arm+'_'+kind]=z;armout[kind]=metrics(z,np.repeat(ys,4) if kind=='background_swap' else ys)
        z=arrays[arm+'_background_swap'].reshape(48,4,2).astype(float);margin=z[:,:,1]-z[:,:,0];rawsd=margin.std(1);scale=tr['arms'][arm]['normalizer'];assert scale>0
        sensitivity=rawsd/scale;prob=1/(1+np.exp(-np.clip(margin,-700,700)));flips=np.any(z.argmax(2)!=z.argmax(2)[:,:1],axis=1)
        arrays[arm+'_sensitivity']=sensitivity;arrays[arm+'_raw_margin_SD']=rawsd
        armout.update(mean_normalized_sensitivity=float(sensitivity.mean()),mean_raw_margin_SD=float(rawsd.mean()),train_only_normalizer=scale,
            any_donor_prediction_flip_rate=float(flips.mean()),mean_probability_range=float(np.ptp(prob,axis=1).mean()),per_source_sensitivity=sensitivity.tolist())
        out[arm]=armout;del m;gc.collect();torch.mps.empty_cache()
    A=arrays['shared_sensitivity']-arrays['semantic_independent_sensitivity'];B=arrays['shifted_independent_sensitivity']-arrays['semantic_independent_sensitivity']
    rng=np.random.default_rng(2026102117);idx=np.concatenate([rng.choice(np.flatnonzero(ys==k),(5000,24),replace=True) for k in [0,1]],axis=1)
    cia=np.quantile(A[idx].mean(1),[.025,.975]);cib=np.quantile(B[idx].mean(1),[.025,.975]);rel=float(A.mean()/arrays['shared_sensitivity'].mean())
    gates=dict(training_valid=True,measurable_shared_sensitivity=out['shared']['mean_normalized_sensitivity']>.001,relative_reduction_at_least20percent=rel>=.2,
        A_CI_lower_positive=bool(cia[0]>0),B_CI_lower_positive=bool(cib[0]>0),
        original_accuracy_drop_no_more5pp=out['semantic_independent']['original']['accuracy']>=out['shared']['original']['accuracy']-.05,
        foregroundonly_accuracy_drop_no_more5pp=out['semantic_independent']['foreground_only']['accuracy']>=out['shared']['foreground_only']['accuracy']-.05)
    np.savez_compressed(ROOT/'heldout-predictions.npz',**arrays)
    save('heldout-pixel-audit.json',dict(rows=audit,originals=48,composites=192,independent_unit='sourcephotograph',heldout_source_IDs_disjoint_prior=True,
        RGB_FG_and_guard_equal_across_backgrounds=True,backgrounds_from_fixed4training_sources=True,no_natural_OOD_or_world_do_claim=True))
    save('final-results.json',dict(status='completed_single_locked_evaluation',at_UTC=now(),arms=out,
        primary_A_mean=float(A.mean()),A_bootstrap95=cia.tolist(),secondary_specificity_B_mean=float(B.mean()),B_bootstrap95=cib.tolist(),relative_primary_reduction=rel,
        candidate_screen=gates,passes_large_effect_pilot_screen=all(gates.values()),
        interpretation='retain_only_as_controlled_replication_signal_not_original_contribution' if all(gates.values()) else 'narrow_predeclared_screen_not_met_no_retuning',
        source_photos=48,training_seed_count=1,background_donors=4,bootstrap_unit='sourcephotosconditionalonfixedmodels/donors',
        actual_world_causal_effect_identified=False,original_method_novelty_established=False,protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(Path(__file__)),
        predictions_sha256=sha(ROOT/'heldout-predictions.npz'),training_freeze_sha256=sha(ROOT/'training-freeze.json')))
    print(json.dumps(load('final-results.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:dict(prepare=prepare,freeze=freeze,train=train,fetch=fetch,evaluate=evaluate_final)[sys.argv[1]]()
    except Exception as e:
        signal.alarm(0)
        if sys.argv[1]=='train':
            for arm in MODELS:
                try:
                    if not (ROOT/(arm+'-step'+str(STEPS[arm])+'.pt')).exists():checkpoint(arm,STEPS[arm],True)
                except Exception:pass
        name='stopped-'+sys.argv[1]+'.json'
        if not (ROOT/name).exists():save(name,dict(at_UTC=now(),type=type(e).__name__,reason=str(e),traceback=traceback.format_exc(),steps=STEPS,resource_checks=RESOURCE))
        raise
