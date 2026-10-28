"""Freeze a training-only, eight-reset-step diagnostic before its execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
PILOT = PROJECT / 'augmentation-research-20261022/pilot-20261022'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert not any((ROOT/n).exists() for n in ['protocol.json','protocol-lock.json','results','analysis.json'])
    manifest=json.loads((PILOT/'fit-manifest.json').read_text())
    train=[r for r in manifest['rows'] if r['candidate_partition']=='train']
    assert len(train)==591
    ordered=sorted(train,key=lambda r:hashlib.sha256(('opt-diagnostic-v1|20261022|'+r['source_id']).encode()).hexdigest())
    selected=[]
    for i,r in enumerate(ordered[:96]):
        selected.append({k:r[k] for k in ['source_id','class_id','candidate_partition','image_sha256','decoded_RGB_sha256','duplicate_cluster_id']})
        selected[-1].update(role='gradient' if i<64 else 'probe',batch=i//16 if i<64 else (i-64)//16)
    assert len({r['source_id'] for r in selected})==96
    cfg={
        'status':'frozen_before_any_new_task_gradient_or_probe_forward',
        'at_utc':datetime.now(timezone.utc).isoformat(),
        'question':'At the original initialization and first warmup AdamW step, how much does front versus back Q/V-LoRA change clean training-only CLS features, relative to gradient-batch variation?',
        'scope':'Optimization/parameterization diagnostic only, not a new augmentation effect, contribution, convergence test, or correction of the original scientific outcome.',
        'model_seed':20261022,'layouts':['front','back'],'rank':4,'blocks':{'front':[0,1,2],'back':[9,10,11]},
        'initialization':'Exactly original shared-by-slot A~N(0,.01^2), B0 and common 37-class random head; compare saved original initial trainable checkpoints.',
        'training_only_selection':'Sort the591 retained training rows by SHA256(opt-diagnostic-v1|20261022|source_id); first64 form4 gradient batches16; next32 form2 disjoint probe batches16. No replacement or class/performance selection.',
        'selected_rows':selected,
        'gradient_transform':'Original weak_pil with epoch0, seed20261022; same pixels/tensors across layouts. No RA condition in this diagnostic.',
        'probe_transform':'Original eval_tensor, but exclusively on32 retained training photos disjoint from gradient batches. No dev/test pixels, logits or accuracy.',
        'optimizer':{'name':'Original AdamW','adapter_base_lr':.0003,'head_base_lr':.001,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'total_original_updates':1110,'logical_step':0,'warmup_updates':111,'adapter_actual_lr':.0003/111,'head_actual_lr':.001/111},
        'resets':'Restore all trainable tensors to identical initialization and create fresh zero-state AdamW before EACH batch. Exactly8 independent one-step updates total; no sequential trajectory, no accumulation.',
        'feature':'Final encoder CLS after encoder LayerNorm, before classifier. Joint original head update cannot directly change this feature; gradient starts from the common initial head.',
        'metrics':{
            'S':'CPU FP64 sqrt(mean((post_step_CLS_FP32 - initial_CLS_FP32)^2)) separately on each probe16 and pooled32.',
            'relative_S':'S / CPU FP64 RMS(initial_CLS)',
            'parameter_diagnostic':'A/B/head gradient and actual parameter-delta norms; CPU FP64 merged Q/V BA norms; first-step AdamW oracle on saved gradients.',
            'batch_variation':'All4 pooled S values per layout; sample SD of log S and CV; paired log(front_S/back_S), geometric-mean ratio and its full range. Two probe batches are sensitivity views, not independent training repeats.',
            'normalization':'Also report S / joint Frobenius norm of merged Q/V update; this does not equalize direction or prove a Jacobian mechanism.'},
        'precision':{'aggregation':'All norms/RMS reported by NEW code use CPU FP64 from saved FP32 tensors; no FP32-versus-FP64 norm equality predicate.',
            'baseline_repeat':'Two identical no-grad initial probe forwards per layout; save both. Compare initial front/back features.',
            'floor':'For each probe max(repeat-forward RMS, eps_float32 * RMS(initial_features)); call one-step S resolved only if S >10*floor. Never increase LR after seeing this check.',
            'initial_front_back_allclose':{'atol':1e-5,'rtol':0},
            'manual_AdamW_parameter_oracle':{'atol':2e-10,'rtol':2e-6,'scope':'First bias-corrected step from saved gradient, including decoupled decay. FP64 mathematical oracle vs FP32 state; fixed tolerance, no post-result relaxation.'},
            'paired_head_gradient_allclose':{'atol':2e-8,'rtol':2e-5}},
        'structural_predictions':['Initial B and merged corrections exact0, initial front/back features match.','Every A data-gradient exactly0 at B0; B gradients finite and nonzero. AdamW decay can still change A.','For first fresh AdamW step, B update follows -eta*g/(abs(g)+eps); parameter gradient norm need not predict AdamW feature-step norm.'],
        'decision':{
            'structural_or_oracle_failure':'Mark diagnostic invalid/incomplete; retain artifacts, do not repair scientific criteria or continue a trajectory.',
            'below_precision_floor':'Report unresolved at the original first warmup step. No larger LR, new seed, batch replacement or extra steps.',
            'consistent_scale_difference':'Only describe if all4 paired pooled ratios and both probes have the same direction and all S are resolved. Report batch variability, no p-value or cross-seed claim.',
            'finish':'Stop after these8 reset steps regardless of outcome. Original pilot engineering gate remains failed; any revised full experiment requires a separate reviewable protocol and is not launched.'},
        'wall_cap_seconds':600,'CPU_threads':1,'serial_MPS_only':True,'MPS_fallback':False,'MPS_recommended_working_set_fraction':.1,'paced_sleep_multiple':2,
        '8A_priority':'Use own small serial process; last parent coordination reported8A finished. No global-idleness assertion or8A content inspection; stop on resource request.',
        'new_dependencies_A100_Colab_ARiS_paid_upload_publish':False,
        'original_pilot_protocol_sha256':sha(PILOT/'protocol.json'),'fit_manifest_sha256':sha(PILOT/'fit-manifest.json'),
        'weight_sha256':sha(PILOT/'data/vit_b_16-c867db91.pth'),
        'protected_original_pilot_delivery_manifest_sha256':sha(PILOT/'delivery-manifest.json'),
        'model_request':{'model':'GPT-6.1-Sol','reasoning_effort':'max','Fast':'existing request retained','actual_backend_receipt_exposed':False}}
    save(ROOT/'protocol.json',cfg)
    save(ROOT/'protocol-lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'protocol.json'),
        'code_sha256':{n:sha(ROOT/n) for n in ['prepare_protocol.py','diagnostic.py','analyze.py']},
        'supporting_files_sha256':{n:sha(ROOT/n) for n in ['protocol-zh.md','primary-sources.json','device-readiness.json','review-zh.md']},
        'original_common_sha256':sha(PILOT/'common.py')})
    print(json.dumps({'status':cfg['status'],'protocol_sha256':sha(ROOT/'protocol.json'),'selected_training_only_photos':96,'planned_reset_updates':8}))

if __name__=='__main__':main()
