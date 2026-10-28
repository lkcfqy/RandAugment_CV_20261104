"""One fixed Equalize pairing experiment. Existing data/weights, CPU only."""
import os
os.environ.update(OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from collections import defaultdict
import datetime
import hashlib
import json
import signal
import sys
import time
import numpy as np
from PIL import Image,ImageOps
import torch
import torchvision
from torchvision.transforms import functional as F

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'encoder-resnet18-round2-20261021'
RUNTIME=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/encoder-readiness-20261021/runtime')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def get_model(protocol):
    assert sha(Path(protocol['head_file']))==protocol['head_file_sha256']
    weight=RUNTIME/'torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
    assert sha(weight)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cpu')
    torch.use_deterministic_algorithms(True)
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    model.fc=torch.nn.Identity();model.eval().requires_grad_(False).to('cpu')
    assert all(p.device.type=='cpu' for p in model.parameters())
    head=json.loads(Path(protocol['head_file']).read_text())['protected']['coupled']
    return model,torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(),head

def scores(features,head):
    z=features.astype(np.float64);z/=np.linalg.norm(z,axis=1,keepdims=True)
    logits=10*z@np.array(head['centers']).T
    return logits[:,1]-logits[:,0]

def infer(images,model,preprocess):
    result=np.empty((len(images),512),dtype=np.float32);start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(images),8):
            batch=torch.stack([preprocess(Image.fromarray(im)) for im in images[i:i+8]]).to('cpu')
            assert batch.device.type=='cpu'
            result[i:i+len(batch)]=model(batch).numpy()
    assert np.isfinite(result).all()
    return result,time.perf_counter()-start

def run(split):
    start=time.perf_counter();protocol=json.loads((ROOT/'protocol.json').read_text())
    if split=='heldout':
        freeze=json.loads((ROOT/'development-freeze.json').read_text())
        assert sha(ROOT/'protocol.json')==freeze['protocol_sha256']
        assert sha(Path(__file__))==freeze['code_sha256']
        assert freeze['hyperparameters_changed_after_development'] is False
    model,preprocess,head=get_model(protocol)
    old=json.loads((OLD/'main-frames.json').read_text())['records']
    raw=np.memmap(OLD/'main-frames.rgba',dtype=np.uint8,mode='r',shape=(len(old),128,128,4))
    selected=[r for r in protocol['selection_rows'] if r['split']==split]
    images=[];records=[];checks=[];examples={};solar_checks=0
    def emit(im,meta):
        idx=len(images);images.append(np.array(im,copy=True));records.append(dict(meta,index=idx,RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
        return idx
    for source in selected:
        case=source['case'];meta={k:source[k] for k in ['case','split','block','foreground_id','Y','cropped']}
        inputs=[r for r in old if r['case']==case and r['layout']=='protected']
        guard=np.asarray(Image.open(ROOT/('guards-'+split)/('case'+str(case)+'.png')).convert('L'))>0
        base=[];equal=[];solar=[]
        for row in inputs:
            rgba=np.array(raw[row['index']],copy=True)
            assert hashlib.sha256(rgba.tobytes()).hexdigest()==row['RGBA_sha256']
            im=Image.fromarray(rgba[:,:,:3].copy())
            base.append(np.array(im));equal.append(np.array(F.equalize(im)));solar.append(np.array(F.solarize(im,128)))
            assert np.array_equal(equal[-1],np.asarray(ImageOps.equalize(im)))
        assert len(inputs)==(16 if split=='fit' else 4)
        assert all(np.array_equal(im[guard],base[0][guard]) for im in base)
        assert all(np.array_equal(im[guard],solar[0][guard]) for im in solar)
        drift=[]
        for j in range(len(inputs)):
            for k in range(len(inputs)):
                negative=solar[j].copy();negative[guard]=solar[k][guard]
                assert np.array_equal(negative,solar[j]);solar_checks+=1
                drift.append(float(np.mean(np.abs(equal[j][guard].astype(np.float64)-equal[k][guard]))))
        indices={}
        for j,row in enumerate(inputs):
            metadata=dict(meta,background_id=row['actual_background_id'],tone_background_id=row['actual_background_id'])
            for kind,im in [('no_op',base[j]),('equalize',equal[j]),('solarize',solar[j])]:
                index=emit(im,dict(metadata,variant=kind))
                indices[j,kind]=index
        if split=='heldout':
            for j,row in enumerate(inputs):
                for k,tone_row in enumerate(inputs):
                    control=equal[j].copy();control[guard]=equal[k][guard]
                    assert np.array_equal(control[~guard],equal[j][~guard])
                    assert np.array_equal(control[guard],equal[k][guard])
                    if j==k:assert np.array_equal(control,equal[j])
                    emit(control,dict(meta,variant='cross_equalize',background_id=row['actual_background_id'],tone_background_id=tone_row['actual_background_id']))
        checks.append(dict(case=case,foreground_id=source['foreground_id'],guard_pixels=int(guard.sum()),
            baseline_guard_identical=True,solarize_guard_identical=True,
            equalize_pairwise_guard_MAE_0_255=float(np.mean(drift)),
            equalize_any_guard_change=any(not np.array_equal(im[guard],equal[0][guard]) for im in equal[1:]),
            all_cross_pairings=len(inputs)**2,processed_guard_and_complement_marginals_exact=True))
        examples[case]=(base,equal,indices)
    features,encoder_seconds=infer(images,model,preprocess)
    with (ROOT/(split+'-features.npy')).open('xb') as f:np.save(f,features,allow_pickle=False)
    ss=scores(features,head);ys=np.array([r['Y'] for r in records])
    loss=np.logaddexp(0,ss)-ys*ss;correct=(ss>0)==ys
    for i,row in enumerate(records):row.update(score_dog_minus_cat=float(ss[i]),loss_nats=float(loss[i]),correct=bool(correct[i]))
    by_source=[]
    for source in selected:
        subset=[r for r in records if r['case']==source['case']]
        table={}
        for kind in sorted(set(r['variant'] for r in subset)):
            rows=[r for r in subset if r['variant']==kind]
            table[kind]={'records':len(rows),'loss':float(np.mean([r['loss_nats'] for r in rows])),
                'accuracy':float(np.mean([r['correct'] for r in rows]))}
        paired_failures=sum(records[examples[source['case']][2][j,'no_op']]['correct'] and not records[examples[source['case']][2][j,'equalize']]['correct'] for j in range(16 if split=='fit' else 4))
        item=dict(source,metrics=table,baseline_correct_equalize_wrong=paired_failures,
            delta_equalize_minus_noop=table['equalize']['loss']-table['no_op']['loss'])
        if split=='heldout':item['delta_equalize_minus_cross']=table['equalize']['loss']-table['cross_equalize']['loss']
        by_source.append(item)
    metrics={}
    for kind in sorted(set(r['variant'] for r in records)):
        values=[s['metrics'][kind] for s in by_source]
        metrics[kind]={'loss':float(np.mean([s['loss'] for s in values])),
            'accuracy':float(np.mean([s['accuracy'] for s in values])),
            'source_count':len(values),'records':sum(s['records'] for s in values)}
    result=dict(status='completed_fixed_'+split+'_equalize_audit',at_utc=now(),split=split,metrics=metrics,
        per_source=by_source,baseline_correct_equalize_wrong=sum(s['baseline_correct_equalize_wrong'] for s in by_source),
        sources_with_changed_guard=sum(c['equalize_any_guard_change'] for c in checks),
        mean_pairwise_guard_MAE_0_255=float(np.mean([c['equalize_pairwise_guard_MAE_0_255'] for c in checks])),
        negative_control_solarize_crosspair_pixel_equality_checks=solar_checks,
        protocol_sha256=sha(ROOT/'protocol.json'),head_sha256=protocol['head_file_sha256'],
        code_sha256=sha(Path(__file__)),CPU_only=True,threads=torch.get_num_threads(),GPU=False,ARIS=False,
        encoder_seconds=encoder_seconds,wall_seconds=time.perf_counter()-start,
        feature_sha256=sha(ROOT/(split+'-features.npy')),preprocess=repr(preprocess))
    if split=='heldout':
        delta=metrics['equalize']['loss']-metrics['cross_equalize']['loss']
        block_delta={str(b):float(np.mean([s['delta_equalize_minus_cross'] for s in by_source if s['block']==b])) for b in range(4)}
        pos_sources=sum(s['delta_equalize_minus_cross']>0 for s in by_source)
        gates=dict(mean_harm_at_least_001=delta>=.01,at_least12of16_sources=pos_sources>=12,
            at_least3of4_blocks=sum(d>0 for d in block_delta.values())>=3,
            global_Equalize_loss_increases_vs_noop=metrics['equalize']['loss']>metrics['no_op']['loss'],
            actual_foreground_guard_pixel_change=result['sources_with_changed_guard']>0)
        result.update(primary_delta_LL_global_minus_cross=delta,block_deltas=block_delta,
            positive_source_count=pos_sources,gates=gates,screen_passed=all(gates.values()))
    save(split+'-image-manifest.json',dict(records=records,pixel_controls=checks,all_images_are_RGB_only=True))
    # Illustrative failures selected only on the development side, after fixed analysis.
    if split=='fit':
        candidates=[]
        for case,(base,equal,indices) in examples.items():
            for j in range(len(base)):
                a,b=records[indices[j,'no_op']],records[indices[j,'equalize']]
                if a['correct'] and not b['correct']:candidates.append((case,j,a,b))
        for n,(case,j,a,b) in enumerate(candidates[:4]):
            before=examples[case][0][j];after=examples[case][1][j]
            canvas=np.concatenate([before,after],axis=1)
            Image.fromarray(canvas).save(ROOT/('development-failure-'+str(n+1)+'.png'))
        save('development-failure-examples.json',dict(source_split='fit',selection='first4 fixed-order baseline-correct to Equalize-wrong development frames; illustration only',examples=[dict(case=t[0],background_index=t[1],before=t[2],after=t[3]) for t in candidates[:4]]))
    save(split+'-results.json',result)
    if split=='fit':save('development-freeze.json',dict(at_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),
        code_sha256=sha(Path(__file__)),development_results_sha256=sha(ROOT/'fit-results.json'),
        hyperparameters_changed_after_development=False,operator_and_head_unchanged=True,
        heldout_not_encoded_in_this_phase=True))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_source','preprocess']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['fit','heldout']
    try:os.nice(10)
    except OSError:pass
    signal.alarm(180)
    run(sys.argv[1])
