#!/usr/bin/env python3
"""Audit fixed LA3 policies with synthetic CPU images; no training or datasets."""
import ast
import collections
import datetime
import hashlib
import io
import itertools
import json
from pathlib import Path
import pickle
import pickletools
import random
import time
import types
import zipfile

import numpy as np
import PIL
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

ROOT = Path(__file__).resolve().parent
GEO = {'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'}
NAMES = ['ShearX','ShearY','TranslateX','TranslateY','Rotate','AutoContrast','Invert','Equalize','Solarize','Posterize','Contrast','Color','Brightness','Sharpness','Cutout','Identity']
BAD_SUFFIX = {'AutoContrast','Equalize','Contrast','Sharpness'}


class BasicOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('object constructors forbidden')
    def persistent_load(self, pid):
        raise pickle.UnpicklingError('persistent references forbidden')


def read_policies():
    output = {}
    allowed = {'PROTO','EMPTY_DICT','BINPUT','MARK','BININT1','BININT2','EMPTY_LIST','BINUNICODE','TUPLE3','BINGET','APPENDS','LONG_BINPUT','LONG_BINGET','SETITEMS','STOP'}
    for name in ['cifar10','cifar100','imgnet']:
        data = (ROOT / f'sources/policies/{name}_policy.pkl').read_bytes()
        assert all(op.name in allowed for op, arg, pos in pickletools.genops(data))
        value = BasicOnly(io.BytesIO(data)).load()
        assert isinstance(value, dict)
        for cls, triples in value.items():
            assert type(cls) is int and isinstance(triples, list)
            assert all(type(t) is tuple and len(t) == 3 and all(type(x) is str and x in NAMES for x in t) for t in triples)
        output[name] = value
    return output


def load_functions():
    path = ROOT / 'sources/code/LA3/augmentations.py'
    tree = ast.parse(path.read_text())
    # Only reviewed pure transform/function definitions; no module imports or training code.
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = {'PIL': PIL, 'np': np, 'random': random, 'random_mirror': True}
    exec(compile(module, str(path), 'exec'), namespace)
    namespace['augment_dict'] = {f.__name__: (f, lo, hi) for f, lo, hi in namespace['augment_list']()}
    return namespace


NS = load_functions()


def actions_for(triple, case, size):
    rr = random.Random(20261029 + case)
    nr = np.random.RandomState(20261029 + case)
    nr.choice(range(101), 1)  # original sampler's policy-choice RNG consumption
    actions = []
    for slot, op in enumerate(triple):
        level = rr.choice(range(10)) if case < 12 else (case - 12) % 10
        py_state, np_state = rr.getstate(), nr.get_state()
        lo, hi = NS['augment_dict'][op][1:]
        physical = lo + level / 10 * (hi - lo)
        sign_draw = rr.random() if op in GEO else None
        if op in GEO and case >= 12:
            # Replay the declared signed geometry within native support.
            wanted_negative = case >= 22
            sign_draw = 0.75 if (physical >= 0) == wanted_negative else 0.25
        center = None
        if op == 'Cutout' and physical > 0:
            # Preserve author's actual np.random.uniform(w), not a repaired range.
            center = [float(nr.uniform(size[0])), float(nr.uniform(size[1]))]
        actions.append({'op':op,'slot':slot,'level':level,'physical':physical,'sign_draw':sign_draw,'center':center,'py_state':py_state,'np_state':np_state})
    return actions


def brief(actions):
    return [{k:v for k,v in a.items() if k not in ['py_state','np_state']} for a in actions]


def apply(img, action):
    if action['sign_draw'] is not None:
        NS['random'] = types.SimpleNamespace(random=lambda: action['sign_draw'])
    if action['center'] is not None:
        draws = iter(action['center'])
        NS['np'] = types.SimpleNamespace(random=types.SimpleNamespace(uniform=lambda *args: next(draws)))
    return NS['apply_augment'](img, action['op'], action['level'] / 10)


def run_chain(canvas, actions, order):
    h, w = canvas.shape[:2]
    img = Image.fromarray(canvas)
    ids = np.arange(1, h*w+1, dtype=np.int32).reshape(h,w)
    kind = np.ones((h,w), dtype=np.int32)  # 1 source, 2 geometry fill, 3 Cutout
    for index in order:
        action = actions[index]
        img = apply(img, action)
        if action['op'] in GEO:
            ids = np.array(apply(Image.fromarray(ids), action), dtype=np.int32)
            kind = np.array(apply(Image.fromarray(kind), action), dtype=np.int32)
            kind[kind == 0] = 2
        elif action['op'] == 'Cutout' and action['physical'] > 0:
            marker = np.asarray(apply(Image.new('RGB', (w,h), (0,0,0)), action))
            mask = np.any(marker != 0, axis=2)
            ids[mask] = 0
            kind[mask] = 3
    return np.asarray(img).copy(), ids, kind


def canvases(size=(32,32)):
    w,h=size; yy,xx=np.mgrid[:h,:w]
    ramp = np.stack([32+(xx*7+yy*3)%192, 32+(xx*3+yy*11)%192, 32+(xx*13+yy*5)%192],2).astype(np.uint8)
    checker = np.stack([((xx//2+yy//2)%2)*255,(xx*255//max(w-1,1)),(yy*255//max(h-1,1))],2).astype(np.uint8)
    return {'constant':np.full((h,w,3),[83,117,161],dtype=np.uint8),'bounded_ramp':ramp,'full_range':checker,'noise':np.random.RandomState(1007).randint(0,256,(h,w,3),dtype=np.uint8)}


def metrics(a, b):
    ia,sa,ka=a; ib,sb,kb=b
    delta=np.abs(ia.astype(np.int16)-ib.astype(np.int16)).max(2)
    same=bool(np.array_equal(sa,sb) and np.array_equal(ka,kb))
    source=(sa>0)|(sb>0)
    nonfill=(ka!=2)|(kb!=2)
    def maximum(mask): return int(delta[mask].max()) if mask.any() else 0
    changed=int(np.count_nonzero(delta))
    exact=bool(same and source.any() and changed>0 and maximum(nonfill)==0)
    tol=bool(same and source.any() and changed>0 and maximum(nonfill)<=1 and maximum(~nonfill)>1)
    reason='strict_fill_only' if exact else ('support_changed' if not same else ('no_source' if not source.any() else ('no_delta' if changed==0 else 'nonfill_changed')))
    return {'same_source_ids_and_channel_masks':same,'source_pixels':int(source.sum()),'fill_pixels_a':int((ka==2).sum()),'fill_pixels_b':int((kb==2).sum()),'cutout_pixels_a':int((ka==3).sum()),'cutout_pixels_b':int((kb==3).sum()),'changed_pixels':changed,'source_changed_pixels':int(np.count_nonzero(delta[source])),'source_max_abs':maximum(source),'non_geometry_fill_max_abs':maximum(nonfill),'geometric_fill_max_abs':maximum(~nonfill),'strict_fill_only':exact,'tolerance_one_fill_only':tol,'status':reason}


def structural_certificate(triple, i, j):
    # Adjacent invert + nearest geometric permutation; optional suffix must not
    # allow fill differences to alter source pixels through statistics/filtering.
    adjacent=abs(i-j)==1
    end=max(i,j)
    return adjacent and (end==2 or triple[2] not in BAD_SUFFIX)


def strict_candidate(triple):
    return len(set(triple)) == 3 and 'Identity' in triple and 'Invert' in triple and sum(x in GEO for x in triple) == 1


def resize_partition(fill, size):
    h,w=fill.shape; ow,oh=size
    def support(n,m):
        basis=np.eye(n,dtype=np.float32)
        weights=np.stack([np.asarray(Image.fromarray(basis[i][None,:]).resize((m,1),Image.Resampling.BICUBIC))[0] for i in range(n)],axis=1)
        return (np.abs(weights)>1e-10).astype(np.int32)
    ay,ax=support(h,oh),support(w,ow)
    has_fill=(ay@fill.astype(np.int32)@ax.T)>0
    has_source=(ay@(~fill).astype(np.int32)@ax.T)>0
    return ~has_fill,~has_source,has_fill&has_source


def postprocess_checks(a,b):
    out=[]
    for top,left in [(0,0),(4,4),(8,8)]:
        arrays=[]
        for pixels,ids,kind in [a,b]:
            p=np.pad(pixels,((4,4),(4,4),(0,0)),constant_values=0)[top:top+32,left:left+32][:,::-1,:].copy()
            # Match the same crop, horizontal flip, and Cutout mask in both arms.
            si=np.pad(ids,4,constant_values=0)[top:top+32,left:left+32][:,::-1].copy()
            ki=np.pad(kind,4,constant_values=4)[top:top+32,left:left+32][:,::-1].copy()
            p[8:24,8:24]=0; si[8:24,8:24]=0; ki[8:24,8:24]=3
            arrays.append((p,si,ki))
        out.append({'pipeline':'CIFAR paired crop/flip/zeroed-Cutout example','top':top,'left':left,**metrics(*arrays)})
    for box in [(0,0,32,32),(4,4,28,28)]:
        x0,y0,x1,y1=box
        pa=a[0][y0:y1,x0:x1];pb=b[0][y0:y1,x0:x1]
        fill=(a[2][y0:y1,x0:x1]==2)
        source,void,mixed=resize_partition(fill,(224,224))
        aa=np.asarray(Image.fromarray(pa).resize((224,224),Image.Resampling.BICUBIC))
        bb=np.asarray(Image.fromarray(pb).resize((224,224),Image.Resampling.BICUBIC))
        delta=np.abs(aa.astype(np.int16)-bb.astype(np.int16)).max(2)
        out.append({'pipeline':'ImageNet-style paired crop/bicubic resize synthetic example','crop_box':box,'pure_source_pixels':int(source.sum()),'pure_fill_pixels':int(void.sum()),'mixed_kernel_support_pixels':int(mixed.sum()),'pure_source_max_abs':int(delta[source].max()) if source.any() else None,'changed_pure_source':int(np.count_nonzero(delta[source])),'changed_mixed':int(np.count_nonzero(delta[mixed])),'changed_pure_fill':int(np.count_nonzero(delta[void])),'mask':'conservative nonzero Pillow float32 bicubic kernel footprint, not an alpha blend estimate'})
    return out


def main():
    results=ROOT/'results';results.mkdir(exist_ok=False)
    policies=read_policies()
    triples=sorted({t for data in policies.values() for rows in data.values() for t in rows if 'Invert' in t and any(x in GEO for x in t)})
    pairs=[]
    for triple in triples:
        for i in range(3):
            if triple[i]!='Invert':continue
            for j in range(3):
                if triple[j] not in GEO:continue
                order=list(range(3));order[i],order[j]=order[j],order[i]
                pairs.append({'pair_id':len(pairs),'triple':triple,'invert_position':i,'geometry_position':j,'order':order,'counterfactual':tuple(triple[k] for k in order),'strict_primary_candidate':strict_candidate(triple),'structurally_certified_adjacent':structural_certificate(triple,i,j)})
    inventory={}
    for name,data in policies.items():
        rows=[]
        for cls,ts in sorted(data.items()):
            counts=collections.Counter(ts)
            eligible=[t for t in ts if 'Invert' in t and any(x in GEO for x in t)]
            cert=[t for t in ts if any(p['triple']==t and p['structurally_certified_adjacent'] for p in pairs)]
            conditions=[]
            for p in pairs:
                if counts[p['triple']]:
                    conditions.append({'pair_id':p['pair_id'],'native_count':counts[p['triple']],'counterfactual_native_count':counts[p['counterfactual']],'native_probability':counts[p['triple']]/len(ts),'counterfactual_probability':counts[p['counterfactual']]/len(ts)})
            rows.append({'class_id':cls,'n_triples':len(ts),'n_unique':len(counts),'eligible_triples':len(eligible),'eligible_probability':len(eligible)/len(ts),'certified_adjacent_triples':len(cert),'certified_adjacent_probability_upper_bound':len(cert)/len(ts),'pair_conditions':conditions})
        inventory[name]={'classes':len(data),'total_triples':sum(map(len,data.values())),'classes_with_eligible':sum(r['eligible_triples']>0 for r in rows),'classes_with_certified_adjacent':sum(r['certified_adjacent_triples']>0 for r in rows),'eligible_triples':sum(r['eligible_triples'] for r in rows),'certified_adjacent_triples':sum(r['certified_adjacent_triples'] for r in rows),'per_class':rows}
    (results/'policy-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (results/'swap-candidates.json').write_text(json.dumps(pairs,indent=2)+'\n')
    # Verify the raw pair identity and default NN/black fill on every native level/sign.
    controls=[]
    for size in [(32,32),(48,32)]:
        for op in sorted(GEO):
            for case in range(12,32):
                for cn,canvas in canvases(size).items():
                    action=actions_for(('Invert',op,'Identity'),case,size)
                    a=run_chain(canvas,action,[0,1,2]);b=run_chain(canvas,action,[1,0,2])
                    controls.append({'size':size,'operation':op,'case':case,'canvas':cn,**metrics(a,b)})
    (results/'operator-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    print(json.dumps({'stage':'inventory_and_operator_controls','pairs':len(pairs),'controls':len(controls),'control_failures':sum(r['status'] not in ['strict_fill_only','no_delta'] for r in controls)}),flush=True)
    summary=[];post=[];started=time.perf_counter();complete=True
    execution_order=sorted(pairs,key=lambda p:(not p['strict_primary_candidate'],p['pair_id']))
    with (results/'all-cases.jsonl').open('x') as allout,(results/'failure-cases.jsonl').open('x') as failures,zipfile.ZipFile(results/'failure-arrays.zip','x',compression=zipfile.ZIP_STORED) as archive:
        for pair in execution_order:
            agg=collections.Counter();max_source=0;max_nonfill=0
            cases=list(range(32)) if pair['strict_primary_candidate'] else [0,1,12,21]
            for case in cases:
                actions=actions_for(pair['triple'],case,(32,32))
                for cn,canvas in canvases().items():
                    if time.perf_counter()-started>180:
                        complete=False;break
                    a=run_chain(canvas,actions,[0,1,2]);b=run_chain(canvas,actions,pair['order'])
                    m=metrics(a,b);key=f"p{pair['pair_id']:03d}-c{case:02d}-{cn}"
                    sse_a=int(np.square(a[0].astype(np.int64)-canvas.astype(np.int64)).sum())
                    sse_b=int(np.square(b[0].astype(np.int64)-canvas.astype(np.int64)).sum())
                    severity={'full_image_sse_to_input_a':sse_a,'full_image_sse_to_input_b':sse_b,'full_image_l2_exact_match':sse_a==sse_b,'rmse_to_input_a':float(np.sqrt(sse_a/canvas.size)),'rmse_to_input_b':float(np.sqrt(sse_b/canvas.size))}
                    row={'case_id':key,'pair_id':pair['pair_id'],'strict_primary_candidate':pair['strict_primary_candidate'],'parameter_case':case,'canvas':cn,'shape':[32,32],'actions':brief(actions),**severity,**m}
                    allout.write(json.dumps(row)+'\n')
                    agg[m['status']]+=1;agg['tested']+=1;agg['tolerance_one_pass']+=int(m['tolerance_one_fill_only'])
                    max_source=max(max_source,m['source_max_abs']);max_nonfill=max(max_nonfill,m['non_geometry_fill_max_abs'])
                    if not m['strict_fill_only']:
                        failures.write(json.dumps(row)+'\n')
                        payload=io.BytesIO();np.savez_compressed(payload,canvas=canvas,a=a[0],b=b[0],source_ids_a=a[1],source_ids_b=b[1],channel_a=a[2],channel_b=b[2])
                        archive.writestr(key+'.npz',payload.getvalue())
                    if case==0 and cn=='bounded_ramp':
                        payload=io.BytesIO();np.savez_compressed(payload,canvas=canvas,a=a[0],b=b[0],source_ids_a=a[1],source_ids_b=b[1],channel_a=a[2],channel_b=b[2])
                        (results/f"representative-{pair['pair_id']:03d}.npz").write_bytes(payload.getvalue())
                        post.append({'pair_id':pair['pair_id'],'precheck':m,'postprocessing':postprocess_checks(a,b) if m['strict_fill_only'] else [],'selection':'fixed case 0 bounded_ramp; no search for a successful seed'})
                if not complete:break
            summary.append({'pair_id':pair['pair_id'],**dict(agg),'max_source_abs':max_source,'max_nonfill_abs':max_nonfill})
            if pair['pair_id']%20==0:print(json.dumps({'pair_finished':pair['pair_id'],'seconds':time.perf_counter()-started}),flush=True)
            if not complete:break
    result={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':'3e593946980912524fb167dad00e83b1b08ec16e','pillow':PIL.__version__,'numpy':np.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'complete':complete,'total_pairs':len(pairs),'strict_primary_pairs':sum(p['strict_primary_candidate'] for p in pairs),'planned_cases':sum((32 if p['strict_primary_candidate'] else 4)*4 for p in pairs),'executed_cases':sum(x['tested'] for x in summary),'core_wall_seconds':time.perf_counter()-started,'pair_summary':summary,'conditions':'fixed synthetic canvases; primary gate limited to Identity/Invert/single-geometry; all other chains are bounded negative/boundary controls; no model or real-image evidence'}
    (results/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (results/'postprocessing.json').write_text(json.dumps(post,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pair_summary'},indent=2),flush=True)


if __name__=='__main__':
    main()
