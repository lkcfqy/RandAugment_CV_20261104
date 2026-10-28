"""Freeze pre-result protocol only after source QA and dummy resource measurement."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
from common import validated_rows
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    manifest=json.loads((ROOT/'fit-manifest.json').read_text())
    preflight=json.loads((ROOT/'preflight.json').read_text())
    assert manifest['status']=='frozen_before_task_fitting'
    validated_rows(manifest)
    assert preflight['status']=='completed_resource_only_calibration'
    assert preflight['augmentation_restores_CPU_and_preserves_MPS_RNG']
    assert all(math.isfinite(r['steady_mean_seconds']) and r['steady_mean_seconds']>0 for r in preflight['cases'])
    assert all(r['device']=='mps:0' and r['trainable_parameters']==65317 for r in preflight['cases'])
    assert json.loads((ROOT/'qa/visual-review.json').read_text())['status']=='completed_before_task_fit'
    assert sha(ROOT/'data/vit_b_16-c867db91.pth')==json.loads((ROOT/'data/weight-receipt.json').read_text())['sha256']
    assert not (ROOT/'results').exists()
    # Resource-only rule, fixed here before dummy measurement or task outcomes.
    predicted=preflight['estimated_30ep_four_cell_seconds_compute_plus_pacing']
    epochs=30 if predicted*1.25<=7200 else 10
    assert predicted*(epochs/30)*1.25<=7200,'Even 10 epochs not feasible within authorized pilot cap.'
    ntrain=sum(r['candidate_partition']=='train' for r in manifest['rows'])
    ndev=sum(r['candidate_partition']=='development' for r in manifest['rows'])
    cfg={'status':'frozen_pre_result_single_seed_development_protocol',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'question':'Does equal-parameter front versus back Q/V-LoRA placement modify the clean-error benefit of fixed RandAugment in this common training recipe?',
        'directional_prediction':'I=(E_front_weak-E_front_RA)-(E_back_weak-E_back_RA)>0',
        'epochs':epochs,'seed':20261022,'batch':16,'wall_cap_seconds':7200,
        'cell_order':[['front','weak'],['back','ra'],['back','weak'],['front','ra']],
        'evaluation_epochs':[e for e in [5,10,20,30] if e<=epochs],
        'endpoint':'Final fixed epoch only. No best epoch, class selection, seed/LR search or final-test reading.',
        'resource_decision':{'rule':'30 epochs if 1.25 times dummy-measured compute+pacing estimate <=7200 seconds; otherwise 10 if feasible, else stop.',
            'dummy_30epoch_estimate_seconds':predicted,'estimated_seconds_with_25percent_overhead':predicted*(epochs/30)*1.25,
            'original_draft_core_epochs':10,'pre_result_revision_reason':'370 updates at 10 epochs may undertrain a random 37-class head. Use the draft maximum30 uniformly if actual local resource cost permits; no task outcome has been inspected.'},
        'architecture':{'name':'torchvision vit_b_16','weights':'IMAGENET1K_V1',
            'head_classes':37,'patch_tokenization':'224x224 RGB, patch16 stride16 ->196 patches plus one class token; same learned position embedding; no text tokenizer.',
            'layout_front':[0,1,2],'layout_back':[9,10,11],
            'LoRA':'Q and V only, r4, alpha/r1, A~N(0,.01^2) shared by slot, B0; K correction identically zero.',
            'trainable_backbone':36864,'trainable_head':28453,'trainable_total':65317,
            'frozen':'All base projection/bias, patch embedding, positional/class token, LayerNorm and all other original parameters.',
            'dropout':0.,'attention_dropout':0.,'precision':'FP32, no AMP'},
        'optimizer':{'name':'AdamW','adapter_lr':.0003,'head_lr':.001,'weight_decay':.01,
            'betas':[.9,.999],'eps':1e-8,'amsgrad':False,'schedule':'10% integer-floor linear warmup then cosine by logical update, as locked common.py.',
            'gradient_clipping':False,'gradient_accumulation':1,'LR_search_runs':0,
            'sensitivity_scope':'A common recipe interaction; equal LR is not proof of equal optimization difficulty or layout-optimal performance. No selective LR repair or extension after seeing the sign.'},
        'augmentation':{'weak':'PIL RRC scale[.8,1] ratio[.9,1.1],224 bilinear antialias; independent HFlip .5.',
            'ra':'Same weak image then torchvision full14-operation RandAugment, N2 M9/31, bilinear, fill[124,116,104].',
            'RNG':'Separate SHA256 keyed CPU RNG for weak and RA by source ID/epoch/seed, source-order key independent of model RNG.',
            'pairing':'Every epoch ordered RGB, normalized FP32 input tensor and source-ID order SHA must match across layouts within each augmentation.'},
        'data':{'train_photos':ntrain,'development_photos':ndev,'source':'Oxford-IIIT Pet fixed mirror revision003aeb4bc80d584ef307dc24e8b38d26cf9f9e43, original trainval only.',
            'reuse':'Each training source once per epoch; same source exposure and logical updates in every cell, no repeated/resampled sources to fill batches.',
            'updates_per_cell':epochs*math.ceil(ntrain/16),'presentations_per_cell':epochs*ntrain,
            'evaluation':'Deterministic shorter-edge256 bilinear antialias, center crop224, ImageNet normalization; argmax takes first maximum, natural-log NLL from FP32 logits in float64.',
            'label_preservation':'Original annotated breed identity is the target; geometry/visual QA supports visibility only. Photometric or crop loss of breed cues can remain; no causal invariance proof.',
            'independence':'Source photograph after conservative known duplicate removal. Unknown animal identity and ImageNet overlap remain unverified.',
            'reserved_confirmation':'592 candidate IDs remain sealed: no reserved members requested by name, extracted, stored, decoded, scored or used for training/selection. Public ZIP range buffers include a discarded header margin; see data/range-transfer-scope.json rather than claiming literal transport-byte exclusion.',
            'license':'Official current CC BY-SA4.0, legacy research/original-site notice preserved. Local research only, attribution/modification notes; no redistribution.'},
        'primary_metric':'source-photo micro classification error on clean development images',
        'secondary_metrics':['macro error','natural-log NLL','all37 per-class errors','front-minus-back absolute baseline gaps','training CE and representation/parameter-change diagnostics'],
        'bootstrap':{'algorithm':'Pair the four cell errors by source ID; independently resample with replacement inside each of37 observed breeds, preserving original breed counts; sum sampled d_i /total N.',
            'replicates':5000,'seed':73102,'generator':'numpy.default_rng PCG64','quantiles':[.025,.975],'method':'numpy.quantile linear',
            'unit':'source photograph after all confirmed/uncertain duplicate clusters removed',
            'scope':'Fixed-model conditional photo-sampling sensitivity only; not training variability, animal-level independence, population coverage or confirmatory inference.'},
        'engineering_gate_I_min':.01,
        'gate_rationale':'One percentage point is the pre-existing proposal engineering investment scale (~6 net error-count units out of592); it is not significance, power, equivalence or a novelty criterion. No CI lower-bound condition is added.',
        'decision':{'I_below_gate':'Retain failure of this common-recipe development gate. No extra seeds to rescue sign, no test opening. Inspect optimization sufficiency honestly; cannot reject all possible position x augmentation effects.',
            'I_above_gate':'Potential development signal only; requires predeclared paired seeds, fair optimization sensitivity and stronger reference arms before an original contribution claim.',
            'both_RA_gaps_negative':'At most moderation of RA harm, not unlocked enhancement benefit.',
            'measurement_failure':'Nonfinite gradients/loss, OOM, missing/frozen/wrong parameters, unequal pixels, budget stop or incomplete endpoints -> incomplete/invalid measurement, not scientific negative.',
            'underfitting':'Report CE and late trajectory without post-hoc numeric convergence threshold. Fixed budget is not a convergence guarantee; do not extend only a weak cell.'},
        'not_run':['spread all12 r1 LoRA','all12 r4 LoRA','full fine-tuning','backup CutMix','additional seeds','LR grid'],
        'claim_boundary':'No SOTA/full-FT superiority, unique mechanism, causal effect identification, publication or teacher-approval guarantee.',
        'resources':{'device':'mps:0','MPS_fallback':False,'CPU_threads':1,'nice_increment':10,
            'MPS_per_process_memory_fraction':.1,'paced_sleep_multiple_of_active_compute':2,
            'actual_utilization_measured':False,'serial_only':True,'8A_priority':True,
            'A100_Colab_ARiS_paid_upload_publish':False,'STOP.request':'Create file in run root for graceful stop after logical update.'},
        'model_request':{'model':'GPT-6.1-Sol','reasoning_effort':'max','Fast':'existing requested configuration retained','actual_backend_receipt_exposed':False},
        'fit_manifest_sha256':sha(ROOT/'fit-manifest.json'),
        'weight_sha256':json.loads((ROOT/'data/weight-receipt.json').read_text())['sha256'],
        'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'qa-intent.json',ROOT/'qa/source-qa.json',ROOT/'qa/geometry-visibility.json',ROOT/'qa/visual-review.json',ROOT/'data/license-receipt.json',ROOT/'data/range-transfer-scope.json',ROOT/'old-id-exclusion-check.json',ROOT/'preflight.json',ROOT/'independent/code-review.md',ROOT/'independent/code-review-closure.md']}}
    save(ROOT/'protocol.json',cfg)
    code=['common.py','train.py','analyze.py','qa.py','preflight.py','make_fit_manifest.py','prepare_protocol.py']
    save(ROOT/'protocol-lock.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(ROOT/'protocol.json'),'code_sha256':{n:sha(ROOT/n) for n in code},
        'task_training_started':False,'reserved_confirmation_read':False})
    print(json.dumps({'status':'protocol_frozen','epochs':epochs,'train':ntrain,'development':ndev,
        'protocol_sha256':sha(ROOT/'protocol.json'),'predicted_seconds':cfg['resource_decision']['estimated_seconds_with_25percent_overhead']}),flush=True)

if __name__=='__main__':main()
