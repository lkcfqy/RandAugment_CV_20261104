"""Write the single pre-result protocol; read old metadata, never feature values."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

WORK = Path(__file__).resolve().parent
PROJECT = Path('/Users/lkc/Downloads/RandAugment_CV_20261104')
ROOT = PROJECT / 'causal-augmentation-research-20261020'
OUT = ROOT / 'causal-resume-20261022'
SOURCE = ROOT / 'norb-view-transfer-pilot-20261021'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    old = json.loads((SOURCE/'protocol.json').read_text())
    previous = json.loads((ROOT/'independent-design-challenge-20261021/experiment-protocol.json').read_text())
    ordinals = {str(c): [obj[1] for obj in old['source_objects'] if obj[0] == c] for c in range(5)}
    states = [{'id': row['state_id'], 'weak_weight': row['weak']} for row in previous['policies'] if row['weak'] != '0']
    p = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': '同一物体/图片/条件支持、同一单因素类别边际、同一无条件完整拍摄条件分布、相同图片权重直方图下，将方位×光照关联在同类物体间一致对齐(B)或抵消(S)，是否使固定弱2D增强池相对clean的开发风险差变化至少+.02nat？',
        'target': 'Finite coefficient-allocation × fixed augmentation-pool interaction. An exploratory local development screen, no established contribution.',
        'source_root': str(SOURCE),
        'source_role': 'source_seen', 'development_role': 'development_seen',
        'data_status': 'All50smallNORB objects already consumed in project. Only source15/development10 used; do not decode prior final25 or sealed pixel/feature files.',
        'input_sha256': {name: sha(SOURCE/name) for name in ('protocol.json','source-feature-bank.npz','normalizer.npz')},
        'coefficients': ['-1','-1/2','0','1/2','1'],
        'control_matrix': [['-1','0','1'], ['-1/2','1/2','0'], ['0','1','-1'], ['1/2','-1','1/2'], ['1','-1/2','-1/2']],
        'object_ordinal_by_class': ordinals,
        'factor_levels': {'azimuth': [0,2,4,6,8,10], 'elevation': [0,4,8], 'lighting': [0,3]},
        'signs': 'sA=-1 for azimuth rank0,1,2, +1 for rank3,4,5; sL=-1 for lighting0, +1 for lighting3. Code groups are not certified semantic front/back or calibrated light doses.',
        'photo_law': 'qU=1/540; qB_k(r)=[1+(4/5)*t[(category+k)%5]*sA*sL]/540; qS_k(r)=[1+(4/5)*C[(category+k)%5,object_ordinal]*sA*sL]/540, k=0..4.',
        'treatments': 'Only alignment of the signed joint-condition weighting coefficients within each class changes. All540 source photos and15 physical objects have positive support. This is weighting, not missing support or additional data acquisition.',
        'matched': ['All unique physical objects, photos, class masses, per-object masses and per-class single-factor marginals',
                    'Unconditional complete azimuth×elevation×lighting joint law',
                    'B/S per-configuration full photo-weight histogram and Kish image-weight N_eff=4500/11, not independent sample size',
                    'Each individual object coefficient multiset across five configurations',
                    'Within each data-law condition the clean/weak arms have identical total original-photo loss weights and label masses'],
        'unmatched_and_identification': 'S contains object×condition associations that cancel within class; B aligns those associations in class. This controlled finite allocation identifies that total intervention contrast, not a uniquely identified pixel-level shortcut or a natural-world do effect. B/S have equal generic row-weight concentration; not all per-class/object conditional weight moments can be matched in each configuration.',
        'states': states,
        'policy_scope': 'Reuse previous low-magnitude five-family pool exactly: identity1/3 plus rotate±5deg,shearX/Y±.05,translateX/Y±9px each1/15. Not canonical full14-operation RA or one shared M; no new strengths chosen after latest null.',
        'label_preservation': 'Y is physical toy identity class; nuisance acquisition A/E/L leaves Y unchanged assuming recorded controlled acquisition. The stored mild affine pool is assumed to retain recognizable class evidence; no blinded semantic annotation was obtained, so this is a conditional readout diagnostic. Tiny prior translation-support loss is not a semantic-label certificate. No synthetic edit is called a physical relighting/3D intervention.',
        'feature_transform': 'Invert old source-clean global mean/scalar RMS to per-image L2 phi, then multiply fixedsqrt512; no fitted new normalization or encoder calls.',
        'readout': {'models': 22, 'arms': ['clean','weak'], 'laws': ['U']+[q+str(k) for k in range(5) for q in ('B','S')],
                    'objective': 'sum_photo q(r) sum_state p(s) CE +.005||W||F²; bias unpenalized',
                    'dtype': 'float64', 'device': 'CPU', 'zero_initialization': True, 'lambda': .01,
                    'LBFGS': {'lr':1., 'max_iter':300, 'max_eval':450, 'history_size':30, 'tolerance_grad':1e-8, 'tolerance_change':1e-12, 'line_search':'strong_wolfe'},
                    'gradient_infinity_gate': 1e-5, 'full_expected_objective': True, 'no_SGD_replicates': True},
        'budget_fairness': 'Unique-data and total expected loss mass matched, not per-photo equality across treatment laws. Clean has540 rows and weak5940 rows; enumerated augmentation expectation is part of treatment. Same optimizer caps/criterion, not identical objective evaluations or exact runtime. Source models solved to same fixed gradient tolerance.',
        'primary': 'For each development object average its36clean views; J*=mean_k{(LLweak−LLclean)_B_k−(LLweak−LLclean)_S_k}; average objects equally within each fixed class.',
        'direction_and_failure': 'Prediction J*>=+.02nat and exhaustive paired object-stability interval lower>0. Failure closes only this fixed setting; negative/reverse or small effect is not rescued by changing alpha,C,metrics,strengths,lambda,encoder,stop or data. No fullRA null inferred.',
        'measurement_gate': 'All22 models finite and grad_inf<=1e-5; Uclean source accuracy>=.8; mean over k clean development accuracy in B and S each>=.4. Numerical/capability failure is not effect falsification.',
        'uncertainty': 'Exactly1024 ordered category-stratified paired resamples: draw two objects with replacement from the two observed in each of five classes. Allviews,k,arms and laws travel together. Stability only, not independent population CI or source-object uncertainty.',
        'reporting': 'Report everyk, all22 absolute clean/weak risks, deltaB,deltaS,deltaU and J*. Five overlapping allocation configurations are not five independent datasets or training seeds. U comparisons descriptive only. J* alone never proves augmentation helps or changes the best policy.',
        'access_order': 'protocol + code + independent-review hashes frozen -> exact metadata matching -> sourcefeatures only -> fit/freeze all22 -> decode development_seen member of source-feature-bank only -> one fixed summary -> independent recomputation',
        'resources': {'CPU_threads':1,'wall_seconds_cap':120,'peak_RSS_bytes_cap':2147483648,'new_images_or_features':False,'MPS_scientific_experiment':False,'A100':False,'Colab':False,'ARIS':False,'8A_access':False,'new_dependencies':False},
        'design_evolution_pre_result': ['Mainline proposed one joint-association question; independent role proposed one real-view redundancy question and rejected its novelty.',
            'Initial uniform-vs-biased comparison had a generic weight-concentration alternative (Kish540 vs409.09). Before any feature-value read this was replaced by the matched-concentration object-balanced control; U retained only descriptively.',
            'Low/high azimuth code groups fixed before new computation. No previous protocol/result altered.'],
        'prior_boundary': [
            {'title':'Wiles etal A Fine-Grained Analysis on Distribution Shift','url':'https://arxiv.org/html/2110.11328v2','read':'§B.2,D.1,D.2,E. Main framework can encode a compound nuisance; smallNORB augmentation/SC already studied. Exact matched-concentration single-factor-balanced comparison not seen in read experiments; not exhaustive.'},
            {'title':'Puli etal Nuisances via Negativa','url':'https://arxiv.org/html/2210.01302','read':'§3 and C.3.2: semantic corruption-powered adjustment and multiple nuisances already studied. Not replicated here; does not establish originality of current exact assignment.'},
            {'title':'Li etal A Whac-A-Mole Dilemma','url':'https://arxiv.org/html/2212.04825v2','read':'§2.1,3,5.2,B.3: multiple-shortcut augmentation harms/last-layer evaluation known; UrbanCars construction uses individually correlated BG/CoObj conditional product. No claim that exact current factorial weighting experiment is fully covered.'}],
        'literature_access_limits': [
            {'target':'https://openreview.net/pdf?id=TMs-EYm4_ms','state':'Historical browser verification, no正文; stopped/no retry or alternate route this round; precise coverage unknown.'},
            {'target':'https://pmc.ncbi.nlm.nih.gov/articles/PMC10923583/','state':'This round browser verification; no正文; stopped/no retry or alternate route. Search snippets not treated as full paper verification.'},
            {'target':'Historical403/restricted sources','state':'No retry, alternate route, or source-code bypass. Prior exclusion/closure table preserved.'}],
        'requested_model': {'model':'GPT-6.1-Sol','reasoning_effort':'max','service_tier':'Fast','actual_backend_receipt_exposed':False,
                            'independent_spawn_explicit_parameters':{'model':'gpt-6.1-sol','reasoning_effort':'max','fork_turns':'none'},'Fast_parameter_on_spawn_exposed':False}
    }
    assert len(states) == 11
    with (OUT/'protocol.json').open('x') as f:
        json.dump(p,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    for name in ('experiment.py','prepare_protocol.py'):
        dest = OUT/name
        assert not dest.exists()
        shutil.copyfile(WORK/name,dest)
    with (OUT/'checkpoint-before-results.md').open('x') as f:
        f.write('''# 因果数据增强恢复：结果前检查点\n\n2026-10-22 UTC。研究继续走因果推断×数据增强，RandAugment是基线；尚无成立的新贡献。没有Git仓库/分支，项目及父目录未找到AGENTS。最新科学回合仍是independent-design-challenge-20261021的阴性；19份字节/SHA与旧回执吻合，旧协议/结果不改。\n\n本轮唯一选定问题：同样物体、图、拍摄条件边际和权重集中程度下，把方位×光照关联在同类物体间对齐或抵消，会否改变固定弱增强池相对clean的风险差。独立角色唯一考虑的真实视图冗余题没有通过增量门。当前问题仅有条件允许一次22头开发测量，不预承认原创。\n\nM5 Pro/64GiB/18逻辑CPU；现有隔离Python3.14.7、torch2.14.1、torchvision0.29.1。MPS真实小矩阵测试0.09748秒且CPU误差0；科学实验使用单线程CPU已有特征，无编码器或图像解码。环境与文件保留依据见environment.json、preservation-baseline.json。\n\nsource15物体×36图、dev10物体×36图全部是已消费开发资料。final25及任何封存像素/数组不解码。共22个固定float64收敛读出，先全部冻结再取dev_seen；主要J*门为+.02nat且对象重采样稳定区间下界>0。失败不改参数或继续试矩阵。详见protocol.json。\n\n已知先例覆盖多捷径、条件移位与增强损害；本轮没有证明当前精确对照全球未做。Geiping验证页边界保留，另PMC原文验证页停止。无A100/Colab/ARIS/新依赖/上传/发布/8A操作。请求Sol/max/Fast；实际后端回执不可得。\n''')
    print(json.dumps({'out':str(OUT),'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(OUT/'experiment.py'),
                      'source_objects':ordinals,'states':[s['id'] for s in states]},ensure_ascii=False))

if __name__ == '__main__':
    main()
