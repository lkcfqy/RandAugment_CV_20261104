"""Staged real-data workflow, dormant until a reviewed release is supplied.

Stage 1 interfaces have no GT or test Y. All generated final Q8 images and
parameters are frozen before stage 2 opens evaluation targets.
"""
import json
from pathlib import Path
import numpy as np
from .core import TRAIN_IDS,TEST_IDS,LEVELS,SEEDS,finite,rgb
from .fit import AParams,BParams,fit_a,fit_b
from .synth import synthesize_a,synthesize_b
from .audit import file_sha,array_sha,save_arrays,load_arrays,expected_forward_keys,validate_records,require_release
from .metrics import psnr,delta_e76
from .diagnostics import diagnose_cell
from .decision import responses,decide,absolute_ranking

PANEL_SHA='5b3da46d5849b0c3b8b71aa9604a87e666523a70e43f19b97a579e042528892b'

class PanelReader:
    def __init__(self,asset_root):
        self.root=Path(asset_root).resolve();path=self.root/'reports/panel.json'
        if file_sha(path)!=PANEL_SHA:raise ValueError('Frozen 84-image panel hash mismatch')
        panel=json.loads(path.read_text());self.entries={}
        for row in panel:
            key=(row['split'],row['scene_id'],row['tag'])
            if key in self.entries:raise ValueError('Duplicate panel identity')
            self.entries[key]=row
        expected={(split,s,t) for split,ids in [('train',TRAIN_IDS),('test',TEST_IDS)] for s in ids for t in ('L10','L1','L3','L5','L7','GT')}
        if set(self.entries)!=expected:raise ValueError('Incomplete panel identities')
        self.evaluation_open=False
    def _read(self,split,scene,tag):
        from PIL import Image
        entry=self.entries[(split,scene,tag)];path=(self.root/entry['local_path']).resolve()
        if not path.is_relative_to(self.root/'data') or file_sha(path)!=entry['raw_sha256']:raise ValueError('Panel image path/hash mismatch')
        with Image.open(path) as im:
            if im.mode!='RGB' or im.size!=(600,400):raise ValueError('Native RGB image required')
            # No ICC handling, EXIF transform, alignment, crop, resize or linearization.
            values=np.array(im,dtype=np.uint8)
        return values.transpose(2,0,1).astype(np.float64)/255
    def training_xy(self,level):
        if level not in LEVELS:raise ValueError('Unknown level')
        return np.stack([self._read('train',s,'L10') for s in TRAIN_IDS]),np.stack([self._read('train',s,'L'+str(level)) for s in TRAIN_IDS])
    def test_x(self,scene):
        if scene not in TEST_IDS:raise ValueError('Unknown test scene')
        return self._read('test',scene,'L10')
    def unlock_evaluation(self,cache):
        validate_generated(cache)
        self.evaluation_open=True
    def evaluation_image(self,split,scene,tag):
        if not self.evaluation_open:raise ValueError('Evaluation targets unavailable before generation freeze')
        return self._read(split,scene,tag)

def save_parameters(stem,p,level,*,budget=None):
    if level not in LEVELS:raise ValueError('Unknown level')
    if isinstance(p,AParams):
        if budget is not None:budget.before_write(p.gain.nbytes+p.beta.nbytes+4096)
        return save_arrays(stem,{'gain':p.gain,'beta':p.beta},{'method':'A','level':level,'gamma':p.gamma,'loss':p.loss})
    if not isinstance(p,BParams):raise ValueError('Unknown parameter type')
    arrays={name:getattr(p,name) for name in ('lut','mu','variance','sigma','original_counts','fallback','spectrum','root')}
    original_bins=np.flatnonzero(p.original_counts).tolist()
    # Empty-bin fallbacks reference their original donor, rather than multiplying
    # the on-disk empirical sample by up to sixteen identical copies.
    arrays.update({f'Q_{b}_{c}':p.quantiles[b][c] for b in original_bins for c in range(3)})
    if budget is not None:budget.before_write(sum(a.nbytes for a in arrays.values())+1024*len(arrays))
    return save_arrays(stem,arrays,{'method':'B','level':level,'iterations':10,'quantile_original_bins':original_bins,'quantile':'ceil(n*u), step, no interpolation','spectrum':'full two-sided ortho; DC=0'})

def load_parameters(stem):
    a,m=load_arrays(stem)
    if m['level'] not in LEVELS:raise ValueError('Unknown parameter level')
    if m['method']=='A':return AParams(float(m['gamma']),a['gain'],a['beta'],float(m['loss']))
    if m['method']!='B' or m['iterations']!=10:raise ValueError('Invalid B parameter metadata')
    if m['quantile_original_bins']!=np.flatnonzero(a['original_counts']).tolist():raise ValueError('Original-bin quantile metadata mismatch')
    quantiles=tuple(tuple(a[f'Q_{int(a["fallback"][b])}_{c}'] for c in range(3)) for b in range(16))
    return BParams(*(a[k] for k in ('lut','mu','variance','sigma')),quantiles,*(a[k] for k in ('original_counts','fallback','spectrum','root')))

def generated_stem(cache,method,scene,level,seed):return Path(cache)/'images'/f'{method}_{scene}_L{level}_seed{seed}'

def generated_keys():return [(m,s,l,seed) for m in ('A','B') for s in TEST_IDS for l in LEVELS for seed in SEEDS]

def validate_generated(cache):
    cache=Path(cache);manifest=json.loads((cache/'generation_manifest.json').read_text())
    expected={tuple(k) for k in generated_keys()}
    if {tuple(r['key']) for r in manifest['images']}!=expected or len(manifest['images'])!=256:raise ValueError('Missing method/scene/level/draw in generated panel')
    if manifest['panel_sha256']!=PANEL_SHA:raise ValueError('Generation panel binding mismatch')
    expected_parameters={f'{m}-L{l}' for m in ('A','B') for l in LEVELS}
    if {r['name'] for r in manifest['parameters']}!=expected_parameters or len(manifest['parameters'])!=8:raise ValueError('Missing method/level parameter cache')
    for record in manifest['images']:
        method,scene,level,seed=record['key'];stem=generated_stem(cache,method,scene,level,seed)
        if file_sha(stem.with_suffix('.npz'))!=record['file_sha256']:raise ValueError('Generated image cache changed')
        if file_sha(stem.with_suffix('.json'))!=record['metadata_sha256']:raise ValueError('Generated metadata changed')
    for rec in manifest['parameters']:
        stem=cache/'parameters'/rec['name']
        if file_sha(stem.with_suffix('.npz'))!=rec['file_sha256'] or file_sha(stem.with_suffix('.json'))!=rec['metadata_sha256']:raise ValueError('Fitted parameter cache changed')
    return manifest

def prepare_generated(reader,cache,*,receipt,freeze_sha,budget):
    require_release(receipt,freeze_sha)
    if receipt.get('scope')!='formal':raise ValueError('Real fit/generation requires formal scope')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    if (cache/'generation_manifest.json').exists():raise ValueError('Generation already frozen; explicit read-only reuse instead of overwrite')
    parameters=[];images=[]
    for level in LEVELS:
        budget.check();x,y=reader.training_xy(level)
        pa=fit_a(x,y);budget.check();pb=fit_b(x,y);budget.check()
        for method,p in [('A',pa),('B',pb)]:
            name=f'{method}-L{level}';stem=cache/'parameters'/name;info=save_parameters(stem,p,level,budget=budget)
            parameters.append({'name':name,'file_sha256':info['file_sha256'],'metadata_sha256':file_sha(stem.with_suffix('.json'))})
        del x,y
        for scene in TEST_IDS:
            source=reader.test_x(scene)
            for seed in SEEDS:
                for method in ('A','B'):
                    budget.check()
                    if method=='A':image=synthesize_a(source,pa,scene,level,seed);extra={}
                    else:image,extra=synthesize_b(source,pb,scene,level,seed)
                    # Lossless final-Q8 cache. Outputs of F are never round-tripped.
                    codes=np.rint(rgb(image,quantized=True)*255).astype(np.uint8)
                    budget.before_write(codes.nbytes+4096)
                    stem=generated_stem(cache,method,scene,level,seed)
                    info=save_arrays(stem,{'rgb_codes':codes},{'method':method,'scene':scene,'level':level,'seed':seed,'source_sha256':array_sha(np.rint(source*255).astype(np.uint8)),'freeze_sha256':freeze_sha,**extra})
                    images.append({'key':[method,scene,level,seed],'file_sha256':info['file_sha256'],'metadata_sha256':file_sha(stem.with_suffix('.json'))})
                    budget.check()
    manifest={'panel_sha256':PANEL_SHA,'freeze_sha256':freeze_sha,'parameters':parameters,'images':images,'test_Y_opened':False,'GT_opened':False,'scientific_forward_count':0}
    budget.before_write(len(json.dumps(manifest).encode('utf-8'))*2)
    (cache/'generation_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    validate_generated(cache);budget.check()
    return manifest

def generated_image(cache,method,scene,level,seed):
    arrays,metadata=load_arrays(generated_stem(cache,method,scene,level,seed))
    if (metadata['method'],metadata['scene'],metadata['level'],metadata['seed'])!=(method,scene,level,seed):raise ValueError('Generated image identity mismatch')
    codes=arrays['rgb_codes']
    if codes.dtype!=np.uint8 or codes.ndim!=3 or codes.shape[0]!=3:raise ValueError('Final uint8 RGB code cache required')
    return codes.astype(np.float64)/255

def evaluate_generated(reader,cache,predict,*,receipt,freeze_sha,budget):
    require_release(receipt,freeze_sha)
    if receipt.get('scope')!='formal':raise ValueError('Evaluation requires formal scope')
    manifest=validate_generated(cache)
    if manifest['freeze_sha256']!=freeze_sha:raise ValueError('Generated panel implementation changed')
    reader.unlock_evaluation(cache);records={};gt_cache={};source_cache={}
    for key in expected_forward_keys():
        budget.check();split,scene,method,level,seed=key
        pair=(split,scene)
        if pair not in gt_cache:gt_cache[pair]=reader.evaluation_image(split,scene,'GT')
        if method=='real':image=reader.evaluation_image(split,scene,'L'+str(level))
        else:image=generated_image(cache,method,scene,level,seed)
        # The inference callable receives exactly one RGB image and no GT.
        output=np.asarray(predict(image));finite(output)
        if output.shape!=image.shape:raise ValueError('Model output shape mismatch')
        output=np.clip(output,0,1)
        records[key]={'psnr':psnr(output,gt_cache[pair]),'delta_e76':delta_e76(output,gt_cache[pair]),'input_sha256':array_sha(np.rint(image*255).astype(np.uint8))}
        budget.check()
    validate_records(records)
    stat={method:{} for method in ('A','B')};bpass=np.zeros((8,4),bool);ident=np.zeros((8,4),bool)
    for j,level in enumerate(LEVELS):
        p=load_parameters(Path(cache)/'parameters'/f'B-L{level}')
        for i,scene in enumerate(TEST_IDS):
            budget.check();source=reader.test_x(scene);real_y=reader.evaluation_image('test',scene,'L'+str(level))
            for method in ('A','B'):
                draws=np.stack([generated_image(cache,method,scene,level,seed) for seed in SEEDS])
                result=diagnose_cell(source,real_y,draws,p,synthetic_degeneracy=receipt['synthetic_degeneracy'])
                stat[method][f'{scene}/L{level}']=result
                if method=='B':bpass[i,j]=result['pass'];ident[i,j]=result['identifiable']
                budget.check()
    assembled={};absolute={};predictions={};real_responses={}
    for metric,color in [('psnr',False),('delta_e76',True)]:
        test_t=np.array([records['test',s,'real',10,None][metric] for s in TEST_IDS]);train_t=np.array([records['train',s,'real',10,None][metric] for s in TRAIN_IDS])
        test_r=np.array([[records['test',s,'real',l,None][metric] for l in LEVELS] for s in TEST_IDS]);train_r=np.array([[records['train',s,'real',l,None][metric] for l in LEVELS] for s in TRAIN_IDS]);predictions[metric]={}
        for method in ('A','B'):
            syn=np.array([[[records['test',s,method,l,seed][metric] for seed in SEEDS] for l in LEVELS] for s in TEST_IDS])
            res=responses(test_t,test_r,syn,train_t,train_r,color=color)
            predictions[metric][method]=res['synthetic']
            predictions[metric]['C_anchor']=res['C_anchor'];predictions[metric]['C_meanD']=res['C_meanD'];real_responses[metric]=res['real']
            if not color:absolute[method]=absolute_ranking(test_r,syn.mean(2),res['real'],eligibility=receipt['absolute_ranking_eligibility'])
        assembled[metric]={'real_D':real_responses[metric].tolist(),'predicted_D':{k:v.tolist() for k,v in predictions[metric].items()},'T_test':test_t.tolist(),'R_test':test_r.tolist(),'T_train':train_t.tolist(),'R_train':train_r.tolist()}
    decision=decide(real_responses['psnr'],real_responses['delta_e76'],predictions['psnr'],predictions['delta_e76'],bpass,c_anchor_confirmed=receipt['c_anchor_confirmed'],identifiable=ident,unidentifiable_policy=receipt['unidentifiable_policy'])
    budget.check()
    return {'forward_count':len(records),'synthetic_images':256,'records':[{'key':list(k),**v} for k,v in records.items()],'response_metrics':assembled,'absolute_R_vs_S_ranking':absolute,'stat_diagnostics':stat,'B_stat_pass':bpass.tolist(),'decision':decision,'control_note':'D and C_anchor use test GT scoring; diagnostic, not a GT-free deploy predictor','compute_spent_seconds':budget.total_spent}
