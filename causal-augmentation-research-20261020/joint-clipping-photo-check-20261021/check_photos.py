"""Single CPU-only exploratory check. Reads prior training photos and frozen models."""
import os
os.environ.update(OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import resource
import signal
import time
import traceback
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as TF, InterpolationMode

ROOT = Path(__file__).resolve().parent
STUDY = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
DATA = STUDY/'mechanism-transfer-pets-20261021'
FITTING = STUDY/'conservative-finetune-repair-20261021'
START = time.perf_counter()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, obj):
    with (ROOT/name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def guard(*_):
    raise TimeoutError('Prespecified wall budget reached; no extension or retuning')


def auc(score, event):
    pos, neg = score[event], score[~event]
    if not len(pos) or not len(neg):
        return None
    pairwise = pos[:,None]-neg[None,:]
    return float(((pairwise>0).astype(np.float64)+0.5*(pairwise==0)).mean())


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device('cpu')
    torch.manual_seed(0)
    signal.signal(signal.SIGALRM, guard)
    signal.alarm(180)
    protocol = json.loads((ROOT/'protocol.json').read_text())
    old_protocol = STUDY/'stage-validity-pilot-20261021/protocol.json'
    rows = json.loads(old_protocol.read_text())['train_rows']
    assert len(rows)==64 and [sum(r['Y']==y for r in rows) for y in (0,1)]==[32,32]
    old_audit = json.loads((FITTING/'input-audit.json').read_text())
    expected = {r['source_id']:r['sha256'] for r in old_audit['training_rows']}
    inputs, manifest = [], []
    for r in rows:
        path = DATA/'photos'/(r['source_id']+'.jpg')
        digest = sha(path)
        assert digest==expected[r['source_id']]
        with Image.open(path) as im:
            im = TF.center_crop(TF.resize(im.convert('RGB'),256,InterpolationMode.BILINEAR,antialias=True),[224,224])
            inputs.append(TF.pil_to_tensor(im).to(torch.float32)/255)
        manifest.append(dict(source_id=r['source_id'],Y=r['Y'],path=str(path),sha256=digest,
                             split_usage='previously fitted training/development photo; not a new test image'))
    x = torch.stack(inputs)
    y = torch.tensor([r['Y'] for r in rows],dtype=torch.long)
    assert x.device.type==y.device.type=='cpu'
    assert x.min()>=0 and x.max()<=1
    save('input-manifest.json',{'rows':manifest,'training_protocol_sha256':sha(old_protocol),
         'prior_photo_audit_sha256':sha(FITTING/'input-audit.json'), 'data_license':str(DATA/'data-license.md')})
    variants = {}
    for name,(exposure,red_gain) in protocol['transforms'].items():
        z = (x*exposure).clamp(0,1)
        z = z*torch.tensor([red_gain,1.,1.])[None,:,None,None]
        variants[name] = z.clamp(0,1)
    red = x[:,0]
    b = ((red>=25/49)&(red<5/7)).float().mean((1,2)).numpy()
    t = (red>=25/49).float().mean((1,2)).numpy()
    s = np.maximum((variants['exposure_only'][:,0]>=1).float().mean((1,2)).numpy(),
                   (variants['illumination_only'][:,0]>=1).float().mean((1,2)).numpy())
    observed_mask = ((variants['exposure_only'][:,0]<1)&(variants['illumination_only'][:,0]<1)&(variants['joint'][:,0]>=1))
    assert torch.equal(torch.from_numpy(b),observed_mask.float().mean((1,2)))
    src = variants['source']
    pred_joint = torch.where(src>0, variants['exposure_only']*variants['illumination_only']/src.clamp_min(1e-30),torch.zeros_like(src)).clamp(0,1)
    reconstruction_max_error = float((pred_joint-variants['joint']).abs().max())
    assert reconstruction_max_error < 1e-6
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    std = torch.tensor([.229,.224,.225])[None,:,None,None]
    model_results, arrays = {}, {'labels':y.numpy(), 'B_joint_only_red_boundary_mass':b,'S_single_red_saturation':s,
                                'T_predicted_total_joint_red_saturation':t}
    for arm, config in protocol['models'].items():
        checkpoint = FITTING/config['file']
        assert sha(checkpoint)==config['sha256']
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert payload['step']==2048 and payload['arm']==arm
        model = torchvision.models.resnet18(weights=None,num_classes=2)
        model.load_state_dict(payload['model'],strict=True)
        model.eval()
        assert all(p.device.type=='cpu' for p in model.parameters())
        del payload
        logits = {}
        with torch.inference_mode():
            for name,z in variants.items():
                outputs = []
                for off in range(0,len(y),8):
                    if time.perf_counter()-START > protocol['resources']['soft_wall_seconds']:
                        guard()
                    outputs.append(model((z[off:off+8]-mean)/std))
                logits[name] = torch.cat(outputs)
                assert bool(torch.isfinite(logits[name]).all())
                arrays[arm+'_'+name+'_logits'] = logits[name].numpy()
            reconstructed_outputs = []
            for off in range(0,len(y),8):
                if time.perf_counter()-START > protocol['resources']['soft_wall_seconds']:
                    guard()
                reconstructed_outputs.append(model((pred_joint[off:off+8]-mean)/std))
            reconstructed_logits = torch.cat(reconstructed_outputs)
            arrays[arm+'_reconstructed_joint_logits'] = reconstructed_logits.numpy()
        pred = {name:z.argmax(1) for name,z in logits.items()}
        correct = {name:p.eq(y) for name,p in pred.items()}
        eligible = correct['source']&correct['exposure_only']&correct['illumination_only']
        event = ~correct['joint'][eligible].numpy()
        def margin(z):
            return z.gather(1,y[:,None]).flatten()-z.gather(1,(1-y)[:,None]).flatten()
        m = -torch.minimum(margin(logits['exposure_only']),margin(logits['illumination_only'])).numpy()
        d = -margin(reconstructed_logits).numpy()
        mask = eligible.numpy()
        scores = {'B_joint_only_red_boundary_mass':b[mask],'S_single_red_saturation':s[mask],
                  'M_single_margin_risk':m[mask],'T_predicted_total_joint_red_saturation':t[mask],
                  'D_direct_reconstructed_joint_margin':d[mask]}
        aucs = {name:auc(score,event) for name,score in scores.items()}
        events, non_events = int(event.sum()),int((~event).sum())
        usable = events>=5 and non_events>=5
        directional = None if any(v is None for v in aucs.values()) else all(aucs['B_joint_only_red_boundary_mass']>aucs[n] for n in ['S_single_red_saturation','M_single_margin_risk','T_predicted_total_joint_red_saturation'])
        model_results[arm] = {'checkpoint_sha256':config['sha256'],
            'training_photo_accuracy_not_generalization':{name:float(v.float().mean()) for name,v in correct.items()},
            'eligible_photo_count':int(mask.sum()),'joint_only_errors':events,'eligible_joint_correct':non_events,
            'eligible_photo_ids':[rows[i]['source_id'] for i in range(len(rows)) if mask[i]],
            'joint_only_error_photo_ids':[rows[i]['source_id'] for i in range(len(rows)) if mask[i] and not bool(correct['joint'][i])],
            'event_count_screen_passed':usable,'AUROC_descriptive_only':aucs,
            'B_exceeds_S_M_T':directional,
            'direct_reconstruction_joint_prediction_disagreements':int((reconstructed_logits.argmax(1)!=pred['joint']).sum()),
            'direct_reconstruction_joint_logit_max_abs_error':float((reconstructed_logits-logits['joint']).abs().max()),
            'single_margin_risk_per_photo':m.tolist()}
        arrays[arm+'_eligible'] = mask
        arrays[arm+'_single_margin_risk'] = m
        print(json.dumps({'arm':arm,'eligible':int(mask.sum()),'joint_only_errors':events,'event_count_screen_passed':usable,
                          'AUC':aucs,'elapsed_seconds':time.perf_counter()-START}),flush=True)
        del model
    np.savez_compressed(ROOT/'predictions.npz',**arrays)
    passed = all(v['event_count_screen_passed'] and v['B_exceeds_S_M_T'] is True for v in model_results.values())
    save('results.json',{'status':'completed_one_prespecified_development_check','independent_validation':False,
         'photo_count':64,'model_count':2,'training_or_fitting':False,'device':'cpu','torch_threads':torch.get_num_threads(),
         'readout_candidates_fixed_before_result':True,'weak_diagnostic_screen_supported':passed,
         'models':model_results,'pixel_predictor':{'joint_reconstruction_max_abs_error':reconstruction_max_error,
            'B_per_photo':b.tolist(),'S_per_photo':s.tolist(),'T_per_photo':t.tolist()},
         'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json'),
         'prediction_archive_sha256':sha(ROOT/'predictions.npz'),'input_manifest_sha256':sha(ROOT/'input-manifest.json'),
         'finished_UTC':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-START,
         'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,
         'limits':['Only previously fitted and explored photos; no generalization or novelty inference',
                   'Encoded-RGB software gains, not observed real exposure or WB interventions',
                   'No new training; not an estimate of augmentation benefit',
                   'A negative or event-limited screen does not refute all cue-aware clipping mechanisms',
                   'No subsequent tuning on these results permitted by this protocol']})
    signal.alarm(0)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        signal.alarm(0)
        save('failure.json',{'status':'failed_or_stopped','type':type(error).__name__,'reason':str(error),
             'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-START,
             'no_parameter_change_or_retry_yet':True})
        raise
