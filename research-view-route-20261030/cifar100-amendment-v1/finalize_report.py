"""Read-only numerical/source audit followed by Chinese report generation; no model forward."""
import hashlib,json,math,statistics,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(name):return json.loads((ROOT/name).read_text())
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def verified_pt(p):
    receipt=json.loads(p.with_suffix('.pt.sha256.json').read_text());assert sha(p)==receipt['sha256']
    return torch.load(p,map_location='cpu',weights_only=True)
lock=read('protocol.lock.json');res=read('results.json');sp=read('splits.json');selection=read('selection.json');source=read('data-validation.json')
ledger=json.loads((BASE/'mps-ledger.json').read_text());assert ledger['active'] is None
assert sha(ROOT/'protocol.lock.json')==read('protocol.lock.sha256.json')['sha256']
for name,digest in lock['implementation_hashes'].items():
    p=Path(name) if Path(name).is_absolute() else ROOT/name;assert sha(p)==digest,('changed locked source',name)
assert sha(ROOT/'selection.json')==read('selection.sha256.json')['sha256']==res['selection_sha256']
assert sha(ROOT/'data'/'original_rgb_uint8.npy')==source['raw_uint8_npy_sha256']
assert sha(Path(source['source_archive']))==source['source_archive_sha256']
rows={r['image_id']:r for r in read('data-manifest.json')}
valy=torch.tensor([rows[i]['class_id'] for i in sp['validation']]);testy=torch.tensor([rows[i]['class_id'] for i in sp['test']])
recipes=['B1','B2','B3','B4','B5','B6','C1','C2'];runs=[];val={r:0 for r in recipes};jval={r:0 for r in recipes};checks=0
lp={x['seed']:x for x in read('lp.json')['seeds']}
for seed in [11,22,33]:
    init=verified_pt(ROOT/'initial'/f'seed{seed}.pt');assert init['model']['head.weight'].shape==(100,192)
    for recipe in recipes:
        folder=ROOT/'runs'/f'seed{seed}-{recipe}';run=json.loads((folder/'complete.json').read_text());runs.append(run)
        assert run['status']=='complete' and run['completed_updates']==lock['recipes'][recipe]['N']
        assert run['initial_state_sha256']==lp[seed]['initial_state_sha256']==init['initial_state_sha256']
        for checkpoint in ['J','final']:
            cp=verified_pt(folder/f'{checkpoint}.pt');assert cp['model']['head.weight'].shape==(100,192)
            assert cp['completed_updates']==(lock['J'] if checkpoint=='J' else lock['recipes'][recipe]['N'])
            p=verified_pt(folder/f'{checkpoint}-validation.pt');assert p['ids']==sp['validation'] and torch.equal(p['labels'],valy)
            assert p['logits'].shape==(1600,100) and p['logits'].dtype==torch.float32 and torch.isfinite(p['logits']).all()
            correct=int((p['logits'].argmax(-1)==valy).sum());assert correct==p['correct'];checks+=1
            if checkpoint=='J':jval[recipe]+=correct
            else:val[recipe]+=correct;assert correct==run['final_validation']['correct']
        if recipe=='B5':
            boundary=verified_pt(folder/'stage1-boundary.pt');assert boundary['stage']==1 and boundary['completed_updates']==lock['recipes']['B5']['N_FT']
        assert sha(folder/'final.pt')==run['final_file_sha256']==selection['final_model_sha256'][f'{seed}-{recipe}']
b=min(recipes[:6],key=lambda r:(-val[r],r));c=min(recipes[6:],key=lambda r:(-val[r],r))
assert (b,c)==(selection['baseline'],selection['candidate'])==(res['baseline'],res['candidate']) and val==selection['val_correct_sum']
testpaths=list((ROOT/'runs').glob('*/test.pt'));assert len(testpaths)==6
paired=[]
for seed in [11,22,33]:
    item={'seed':seed}
    for recipe in [b,c]:
        p=verified_pt(ROOT/'runs'/f'seed{seed}-{recipe}'/'test.pt')
        assert p['ids']==sp['test'] and torch.equal(p['labels'],testy) and p['logits'].shape==(5000,100)
        assert p['logits'].dtype==torch.float32 and torch.isfinite(p['logits']).all()
        correct=int((p['logits'].argmax(-1)==testy).sum());assert correct==p['correct'];item[recipe]=correct;checks+=1
    item['delta_correct']=item[c]-item[b];item['delta_pp']=float(Fraction(100*item['delta_correct'],5000));paired.append(item)
mean=Fraction(100*sum(p['delta_correct'] for p in paired),3*5000);gate=mean>=1 and all(p['delta_correct']>0 for p in paired)
assert gate==res['usefulness_gate_passed'] and abs(float(mean)-res['mean_delta_pp'])<1e-12
assert all(abs(p['delta_pp']-next(q['delta_pp'] for q in res['paired_test'] if q['seed']==p['seed']))<1e-12 for p in paired)
assert ledger['cumulative_seconds']<=7200 and time.time()<datetime(2026,10,31,1,14,tzinfo=timezone.utc).timestamp()
disk=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file());assert disk<3*1024**3
audit={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'formal_runs':len(runs),'prediction_files_checked':checks,'selected_test_files':len(testpaths),
       'global_selection_reproduced':True,'usefulness_gate_recomputed_with_exact_integer_counts':True,'all_shared_initializations_verified':True,
       'all_locked_source_and_split_hashes_verified':True,'all_final_J_and_B5_boundary_weight_receipts_verified':True,'original_archive_and_uint8_data_hashes_unchanged':True,
       'mean_delta_pp_fraction':str(mean),'mean_delta_pp':float(mean),'usefulness_gate_passed':gate,'mps_stage_total_seconds':ledger['cumulative_seconds'],
       'known_new_tree_bytes':disk,'validation_test_model_work_cpu':False,'only_cpu_metrics_and_serialization':True}
dump(ROOT/'final-audit.json',audit)
names={'B1':'weakFT','B2':'dualFT','B3':'dual低LR','B4':'EntAug短迁移适配','B5':'weakFT→head-refit','B6':'随机路由','C1':'候选','C2':'候选低LR'}
outcome='满足预设实用继续门' if gate else '未达到预设实用继续门'
lines=['# CIFAR-100 视图梯度路由有限筛选报告','',f'**{outcome}。** 全部24个正式运行与最终6模型测试完成。最终验证选择基线 **{b}（{names[b]}）**、候选 **{c}（{names[c]}）**。候选相对基线平均测试差为 **{float(mean):+.4f} 个百分点**。该门是工程投入筛选，不是显著性检验，也未确立原创性。','',
'## 预结果修订与来源','',
'Pets 官方资产访问失败；安全模型检查通过，但真实 LP/校准/效用均为0。主线在任何真实效用前传达 Pro 审过的一次资源域修订，登记于 `AMENDMENT.json`。原 Pets 协议和报告保留在父目录，关闭时状态/账本快照保留在 `pets-close-snapshot/`。本轮没有再换域，没有借旧 TTA-CP 结果或其划分选数据。','',
f'官方 CIFAR-100 原始归档只读复用，MD5 `{source["source_archive_md5"]}`，SHA256 `{source["source_archive_sha256"]}`；来源为[官方 CIFAR 页面](https://www.cs.toronto.edu/~kriz/cifar.html)。原始数据解析没有执行 pickle globals，只保存32×32 uint8数组，不缓存放大图片。使用 fine labels 0..99。','',
'模型为官方非蒸馏 DeiT tiny、timm 1.0.22；先安全加载完整1000类模型，再换100类线性head。原始[官方权重](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)完整SHA256为 `a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`。MPS FP32，无AMP/compile/dropout/stochastic depth或CPU模型fallback。','',
'## 数据与标签预算','',
f'原官方train50000、test10000。按精确原始record及宽高+RGB hash处理{source["duplicate_groups"]}个重复组：标签冲突隔离{source["exclusion_counts"].get("label_conflict",0)}张，另排除{source["exclusion_counts"].get("duplicate",0)}张重复。清理后train {source["included_counts"]["trainval"]}、test {source["included_counts"]["test"]}。跨split重复保留test代表；没有人工近重复挑图。','',
'每类16support（1600/seed，seeds11/22/33）与另16公共validation（1600）；16-shot只指梯度标签，验证标签需额外披露。test按预注册hash每类取50，共5000，其余test不使用。预处理和hash所需的原始test文件在选模前已读取，test效用只在selection锁后计算。','',
f'划分SHA256：`{source["splits_sha256"]}`；逐图manifest SHA256：`{source["manifest_sha256"]}`。各臂同seed共用support及一次200更新的LP初始化，初始化状态hash已逐run核验。LP/val/test直接32→224 bicubic；训练在原32²上RRC→224+flip，强图从同weak副本做原生RA(2,M9)。','',
'## 冻结预算与运行结果','',
f'正式前真实校准通过。锁定protocol SHA256 `{sha(ROOT/"protocol.lock.json")}`。J={lock["J"]}仅为共同步数的次要验证留档，主选择只用最终验证；无中途补训练、改N/LR/seed或删基线。','',
'| 配方 | 锁定N | J验证均值% | 最终验证均值% | 实际训练循环均值秒 | 完整run均值秒 |','|---|---:|---:|---:|---:|---:|']
for recipe in recipes:
    rr=[r for r in runs if r['recipe']==recipe];cfg=lock['recipes'][recipe];ntext=f'{cfg["N_FT"]}+{cfg["N_head"]}' if recipe=='B5' else str(cfg['N'])
    lines.append(f'| {recipe} {names[recipe]} | {ntext} | {100*jval[recipe]/4800:.4f} | {100*val[recipe]/4800:.4f} | {statistics.mean(r["loop_seconds"] for r in rr):.3f} | {statistics.mean(r["full_run_seconds"] for r in rr):.3f} |')
lines+=['','每run名义训练循环目标180秒，B5名义144秒FT+36秒head。步数由校准速度取整固定，实际完整run还包括加载、增强、同步、检查点、J及final验证；表中完整run计时截止final验证结束；随后记录封存、状态hash和模型释放等收尾仍计入共用MPS总账。本设计是近似等时间，不能称秒级完全相等。B5保留head权重并重建阶段2 optimizer/schedule；阶段边界权重已保存。B5三个seed的J=512都位于stage1弱图微调阶段（阶段1共1251步）。','',
'## 最终选择及封存测试','',f'selection SHA256：`{sha(ROOT/"selection.json")}`。按跨3seed最终验证整数正确总数全局选 {b}/{c}，并列按ID；不是per-seed选择。仅这两配置×3seed共6个模型开启test。','',
f'| Seed | {b}正确/5000 | {b} Acc% | {c}正确/5000 | {c} Acc% | 配对Δ pp |','|---|---:|---:|---:|---:|---:|']
for p in paired:lines.append(f'| {p["seed"]} | {p[b]} | {p[b]/50:.4f} | {p[c]} | {p[c]/50:.4f} | {p["delta_pp"]:+.4f} |')
lines+=['',f'平均配对Δ = {float(mean):+.4f}pp；三seed严格正差 = {all(p["delta_correct"]>0 for p in paired)}；平均≥1pp = {mean>=1}。**{outcome}。** 门计算已以整数正确数重算，防止浮点边界改变判断。','',
'| 选中配置 | Seed | 实际训练循环秒 | 实际完整run秒 |','|---|---:|---:|---:|']
for recipe in [b,c]:
    for r in sorted([r for r in runs if r['recipe']==recipe],key=lambda x:x['seed']):lines.append(f'| {recipe} | {r["seed"]} | {r["loop_seconds"]:.3f} | {r["full_run_seconds"]:.3f} |')
lines+=['','## 验证、资源与结论范围','',
'37类原检查与100类补充检查均通过实际DeiT MPS独立autograd oracle；候选骨干系数是gw，head是(hw+hs)/2。只截断强分支对骨干的直接梯度，head仍可影响后续骨干训练。EntAug使用旧cache的置信度标量、一次forward后更新，属于独立短迁移适配，未声称复现原论文。','',
f'最终核验通过：24个run的完成步数、共享初始化、48份J/final验证预测、6份test预测及所有对应权重hash均匹配；B5阶段边界保留。正式优化器更新总数{sum(r["completed_updates"] for r in runs)}。无CPU模型计算；审计只在CPU读取权重/预测并计算指标。','',
f'共用MPS累计阶段时间 **{ledger["cumulative_seconds"]:.3f}秒 / 7200秒**（含原Pets {ledger["phases"][0]["elapsed_seconds"]:.3f}秒、100类oracle、LP、校准、正式训练/评估与进程收尾）。原wall起点2026-10-30 19:14UTC，截止2026-10-31 01:14UTC，未重置。审计时新增研究目录{disk/1024**3:.4f}GiB；监督器观察峰值{ledger["peak_new_bytes"]/1024**3:.4f}GiB，低于3GiB。已有CIFAR archive不复制、不改写。','',
'科学状态与资源状态分开：本次有限配方效用已完成评估；没有OOM/不支持算子救援，也未改精度、batch、臂或门。没有Colab、付费、OAuth、commit/push或公开上传。','',
'结论只限此固定CIFAR-100清理划分、每类16support加16validation、三个support seed与短迁移预算；不外推Pets、充分收敛训练或其他数据域。预训练数据重叠没有额外排除。三个seed的实用门不构成统计显著性或独立数据集复现。','',
'Deep Augmentation、APNetwork、Gradient Routing 的既有工作限制了新原则和原创性主张。本任务定位为经验配方筛选。'+('该结果只支持另行审查下一轮与直接prior的比较；本轮不自动扩展数据或新实验。' if gate else '该候选未通过预设投入门，本轮结束，不再调整baseline/LR/seed或换域挽救结果。'),'',
'机器可读结果：`results.json`、`selection.json`、`final-audit.json`、`protocol.lock.json`、`data-validation.json`、`../mps-ledger.json`。逐run永久权重、预测、配置、训练轨迹与hash在 `runs/`。']
(ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
dump(BASE/'CURRENT_RESULT.json',{'protocol':'cifar100_view_route_pilot_v1','status':'complete','report':str(ROOT/'REPORT_ZH.md'),'usefulness_gate_passed':gate,'mean_test_delta_pp':float(mean),'Pets_status':'asset_blocked_before_utility','one_domain_amendment_only':True,'mps_seconds':ledger['cumulative_seconds']})
print(json.dumps(audit,ensure_ascii=False,indent=2))
