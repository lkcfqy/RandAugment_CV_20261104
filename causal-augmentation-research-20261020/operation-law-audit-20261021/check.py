"""Frozen, CPU-only augmentation law audit; no fitting, model weights or test photos."""
import os
import sys
import time

START = time.perf_counter()
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
# A setup attempt to lower priority via os.nice(10) was denied by the OS.
# Do not retry or elevate it. Single CPU thread, ordinary priority, short audit only.

import hashlib
import json
import resource
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021')
OLD_PROTOCOL = STUDY / 'brightness-coupling-training-20261021/protocol.json'
PHOTOS = STUDY / 'mechanism-transfer-pets-20261021/photos'
TV_SOURCE = STUDY / 'encoder-readiness-20261021/runtime/.venv/lib/python3.14/site-packages/torchvision/transforms/autoaugment.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def setup():
    old = json.loads(OLD_PROTOCOL.read_text())
    selected = []
    for label in (0, 1):
        rows = [r for r in old['train_rows'] if r['Y'] == label][:4]
        assert len(rows) == 4
        for r in rows:
            path = PHOTOS / (r['source_id'] + '.jpg')
            data = path.read_bytes()
            selected.append({'source_id':r['source_id'], 'Y':label, 'path':str(path),
                             'bytes':len(data), 'sha256':sha(data),
                             'role':'previously used training/development photo; not independent validation'})
    protocol = {
        'at_UTC':datetime.now(timezone.utc).isoformat(),
        'question':'Does the installed 14-operation, 31-bin TrivialAugmentWide produce the same class-independent fill image through its maximal translation on 32x32 inputs?',
        'claim_status':'Implementation/assumption diagnostic, not a novel method, natural-OOD performance claim or causal-effect estimate.',
        'selection':'First four rows per class in the previously frozen 64-photo train_rows, without reading any held-out photo or choosing by current pixels/scores.',
        'source_train_protocol':str(OLD_PROTOCOL),
        'source_train_protocol_sha256':sha(OLD_PROTOCOL.read_bytes()),
        'photos':selected,
        'resolutions':[32,96,224],
        'resize':'RGB, square bilinear resize solely for an operator mechanism check; not an evaluation of Pets at these sizes.',
        'operators':['TranslateX','TranslateY'],
        'bins':31,
        'levels':list(range(31)),
        'signs':[-1,1],
        'fill_default':0,
        'endpoint_fill_control':128,
        'interpolation':'NEAREST, matching installed default',
        'forced_event':'op in {TranslateX,TranslateY}, level=30, either sign; magnitude=32. On32x32 every original pixel is outside output canvas.',
        'event_mass_exact':'2/14 * 1/31 = 1/217, not the total constant-image atom mass.',
        'augmented_risk_only':'For class-independent forced erasure and prior pi, augmented 0-1 Bayes risk >= q*(1-max(pi)); cross entropy >= q*H(pi). No clean/OOD-risk implication.',
        'negative_control':'At96/224 the same maximal translation leaves original support in frame; do not claim all semantic information or all other paths remain safe.',
        'additional_checks':['8bitLUT injectivity for Posterize and Solarize; Invert squared exact identity',
                             'source+fixed edit kernel does not identify an unspecified natural-target classifier-risk difference'],
        'software_source':str(TV_SOURCE),
        'software_source_sha256':sha(TV_SOURCE.read_bytes()),
        'no_training':True,'no_weights':True,'no_test_photos':True,
        'GPU':False,'MPS':False,'A100':False,'ARIS':False,
        'CPU_threads':1,'priority':'ordinary; no OS priority change after denied os.nice attempt',
        'downloads':False,'dependency_changes':False,
    }
    write_new('protocol.json',protocol)
    write_new('protocol-lock.json',{
        'at_UTC':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(Path(__file__).read_bytes()),
        'protocol_sha256':sha((ROOT/'protocol.json').read_bytes())})
    print(json.dumps({'status':'locked_before_pixel_execution','photos':len(selected),'levels':31,'CPU_only':True}))


def check():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    lock = json.loads((ROOT/'protocol-lock.json').read_text())
    assert lock['code_sha256']==sha(Path(__file__).read_bytes())
    assert lock['protocol_sha256']==sha((ROOT/'protocol.json').read_bytes())
    assert protocol['software_source_sha256']==sha(TV_SOURCE.read_bytes())
    import numpy as np
    import PIL
    from PIL import Image, ImageOps
    import torch
    import torchvision
    from torchvision.transforms import TrivialAugmentWide, RandAugment, InterpolationMode
    from torchvision.transforms.autoaugment import _apply_op
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ta = TrivialAugmentWide()
    space = ta._augmentation_space(31)
    assert len(space)==14
    assert ta.interpolation == InterpolationMode.NEAREST
    for name in protocol['operators']:
        mags,signed = space[name]
        assert len(mags)==31 and signed and float(mags[-1])==32.0

    rows=[]
    endpoint_hashes={}
    for item in protocol['photos']:
        data=Path(item['path']).read_bytes()
        assert sha(data)==item['sha256']
        with Image.open(item['path']) as source:
            original=source.convert('RGB')
        for size in protocol['resolutions']:
            img=original.resize((size,size),resample=Image.Resampling.BILINEAR)
            inp=np.asarray(img)
            assert np.unique(inp.reshape(-1,3),axis=0).shape[0]>1
            kept=[]
            constants=0
            for op in protocol['operators']:
                magnitudes,_ = space[op]
                for level in protocol['levels']:
                    for sign in protocol['signs']:
                        m=float(magnitudes[level])*sign
                        out=_apply_op(img,op,m,InterpolationMode.NEAREST,fill=None)
                        arr=np.asarray(out)
                        all_fill=bool(np.all(arr==0))
                        constants+=all_fill
                        # This is a support mask, not a foreground/background manipulation.
                        support=_apply_op(Image.new('L',(size,size),255),op,m,InterpolationMode.NEAREST,fill=0)
                        visible=int(np.count_nonzero(np.asarray(support)))
                        if level==30:
                            erases=(visible==0)
                            assert erases==(size==32)
                            if size==32:assert all_fill
                            out128=_apply_op(img,op,m,InterpolationMode.NEAREST,fill=128)
                            if size==32:assert np.all(np.asarray(out128)==128)
                            key=(size,op,sign)
                            endpoint_hashes.setdefault(key,set()).add(sha(arr.tobytes()))
                            kept.append({'op':op,'sign':sign,'magnitude':m,
                                         'all_default_fill':all_fill,'visible_input_pixels':visible,
                                         'visible_input_fraction':visible/(size*size),
                                         'raw_RGB_sha256':sha(arr.tobytes())})
            rows.append({'source_id':item['source_id'],'Y':item['Y'],'size':size,
                         'tested_translate_states':124,
                         'constant_default_fill_count_in_translate_states':constants,
                         'constant_count_not_a_full_policy_probability':True,
                         'endpoints':kept})
    assert all(len(h)==1 for key,h in endpoint_hashes.items() if key[0]==32)
    # All 32px endpoints coincide across both classes, axes and signs.
    all32={next(iter(h)) for key,h in endpoint_hashes.items() if key[0]==32}
    assert len(all32)==1
    ramp=Image.fromarray(np.arange(256,dtype=np.uint8)[None,:])
    luts=[]
    for bits in (0,2,4,8):
        out=np.asarray(ImageOps.posterize(ramp,bits)).ravel()
        luts.append({'op':'Posterize','parameter':bits,'distinct_outputs':len(set(map(int,out))),
                     'injective':len(set(map(int,out)))==256})
    for threshold in (0,128,255,256):
        out=np.asarray(ImageOps.solarize(ramp,threshold)).ravel()
        count=len(set(map(int,out)))
        assert count==max(threshold,256-threshold)
        luts.append({'op':'Solarize','parameter':threshold,'distinct_outputs':count,'injective':count==256})
    assert np.array_equal(np.asarray(ImageOps.invert(ImageOps.invert(ramp))),np.asarray(ramp))
    q=Fraction(2,14*31)
    risk_floor=q*Fraction(1,2)
    import math
    # Independent small enumerated example: identical source/edit observations, opposite target-risk preference.
    source=[]
    target_risks=[]
    for flipV in (False,True):
        r0=r1=Fraction(0)
        for y in (-1,1):
            for sign,prob in ((1,Fraction(9,10)),(-1,Fraction(1,10))):
                u=y*sign;v=-y if flipV else y
                r0+=Fraction(1,2)*prob*(v!=y)
                r1+=Fraction(1,2)*prob*(u!=y)
                if not flipV:source.append({'Y':y,'U':u,'V':v,'probability':str(Fraction(1,2)*prob)})
        target_risks.append({'V_mechanism':'-Y' if flipV else 'Y','R_h0':str(r0),'R_h1':str(r1),'difference_h1_minus_h0':str(r1-r0)})
    assert target_risks[0]['difference_h1_minus_h0']=='1/10'
    assert target_risks[1]['difference_h1_minus_h0']=='-9/10'
    result={
        'status':'planned_operator_and_logical_checks_passed',
        'at_UTC':datetime.now(timezone.utc).isoformat(),
        'code_sha256':lock['code_sha256'],'protocol_sha256':lock['protocol_sha256'],
        'versions':{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'Pillow':PIL.__version__,'numpy':np.__version__},
        'TA_op_names':list(space),'TA_bins':31,'signed_translations':True,
        'enumerated_photo_translation_states':len(rows)*124,
        'apply_op_calls_photo_plus_support_plus_endpoint_fill_control':len(rows)*(124*2+4),
        'source_photo_count':8,'source_classes':[0,1],'photos_previously_used_training_only':True,
        'forced_32px_erasure_probability_exact':str(q),'probability_decimal':float(q),
        'all_32px_endpoint_RGB_outputs_cross_label_identical':True,
        'all_32px_endpoint_RGB_sha256':next(iter(all32)),
        'augmented_01_risk_lower_bound_balanced_binary':str(risk_floor),
        'augmented_01_risk_lower_bound_balanced_binary_decimal':float(risk_floor),
        'augmented_CE_risk_lower_bound_balanced_binary_nats':float(q)*math.log(2),
        'bound_is_NOT_clean_or_OOD_risk':True,
        'bound_q_is_NOT_total_atom_probability':True,
        'constant_fill_positive_control_128_passed':True,
        'larger_resolution_control':'The specified full-canvas-erasure translation event is absent at96/224; no global semantic-safety claim.',
        'per_photo_translation_checks':rows,
        'LUT_injectivity':luts,'invert_twice_exact_identity':True,
        'RandAugment_default_M9_posterize_bits':int(RandAugment()._augmentation_space(31,(32,32))['Posterize'][0][9]),
        'source_target_nonidentifiability_check':{'same_source_distribution_in_both_worlds':source,'same_any_fixed_source_edit_kernel':True,'targets':target_risks,'not_an_empirical_CV_result':True},
        'new_method_or_contribution_established':False,'actual_model_training':False,
        'wall_seconds_including_imports':time.perf_counter()-START,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'CPU_threads':torch.get_num_threads(),'GPU':False,'MPS':False,'A100':False,'ARIS':False,
        'downloads':False,'dependency_changes':False,'heldout_photos_opened':0,
    }
    write_new('results.json',result)
    print(json.dumps({key:result[key] for key in ('status','source_photo_count','enumerated_photo_translation_states','forced_32px_erasure_probability_exact','augmented_01_risk_lower_bound_balanced_binary_decimal','augmented_CE_risk_lower_bound_balanced_binary_nats','wall_seconds_including_imports','peak_RSS_bytes','heldout_photos_opened')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    {'setup':setup,'check':check}[sys.argv[1]]()
