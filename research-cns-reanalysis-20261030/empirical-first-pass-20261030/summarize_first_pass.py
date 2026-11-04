"""Summarize fixed existing outputs; no downloads, inference, or new endpoints."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'empirical-keymatched-001'
MODELS = ('resnet18.a1_in1k', 'vit_base_patch16_224.augreg_in1k')
NAMES = dict(zip(MODELS, ('ResNet-18', 'ViT-B AugReg')))
SHIFTS = ('cartoon_style', 'heavy_snow')


def read_csv(name):
    with (OUT / (name + '.csv')).open(newline='') as f:
        return list(csv.DictReader(f))


def write_new(path, text):
    with path.open('x') as f:
        f.write(text)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    status = json.loads((OUT / 'status.json').read_text())
    source = json.loads((OUT / 'source_bootstrap.json').read_text())
    cluster = json.loads((OUT / 'class_cluster_bootstrap.json').read_text())
    point = dict(zip(source['coordinate_order'], source['point']))
    intervals = {
        label: {boot['coordinate_order'][i]: bounds for i, bounds in
                zip(boot['primary_coordinate_indices'], boot['simultaneous_primary_intervals'])}
        for label, boot in [('source', source), ('cluster', cluster)]
    }
    summary_rows = []
    for model in MODELS:
        for metric in ('D_C', 'D_E'):
            key = model + '/' + metric
            summary_rows.append({'evidence_type': status['evidence_type'], 'model': model,
                                 'metric': metric, 'estimate_pp': point[key] * 100,
                                 'source_joint95_low_pp': intervals['source'][key][0] * 100,
                                 'source_joint95_high_pp': intervals['source'][key][1] * 100,
                                 'cluster_joint95_low_pp': intervals['cluster'][key][0] * 100,
                                 'cluster_joint95_high_pp': intervals['cluster'][key][1] * 100,
                                 'strict_source_identity_verified': False})
    with (OUT / 'primary_change_summary.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    features = read_csv('trajectory_features')
    indexed = {m: {s: {} for s in SHIFTS} for m in MODELS}
    for row in features:
        indexed[row['model']][row['shift']][(int(row['class_id']), int(row['seed']))] = row
    common = {m: set(indexed[m][SHIFTS[0]]) & set(indexed[m][SHIFTS[1]]) for m in MODELS}
    eligible = {m: {k for k in common[m] if all(indexed[m][s][k]['first_observed_correct'] == 'True'
                                               for s in SHIFTS)} for m in MODELS}
    cross = set.intersection(*eligible.values())
    transitions = []
    for m in MODELS:
        for s in SHIFTS:
            cohorts = {'publisher_complete': set(indexed[m][s]),
                       'common_all_firstpoint_states': common[m],
                       'common_first_observed_correct': eligible[m],
                       'cross_model_common_first_observed_correct': cross}
            for cohort, keys in cohorts.items():
                rows = [indexed[m][s][k] for k in keys]
                transitions.append({'evidence_type': status['evidence_type'], 'model': m, 'shift': s,
                    'cohort': cohort, 'n': len(rows),
                    'all_correct': sum(r['all_correct'] == 'True' for r in rows),
                    'first_observed_errors': sum(r['first_observed_correct'] == 'False' for r in rows),
                    'endpoint_correct': sum(r['endpoint_correct'] == 'True' for r in rows),
                    'correct_to_wrong_transitions': sum(int(r['correct_to_wrong']) for r in rows),
                    'wrong_to_correct_transitions': sum(int(r['wrong_to_correct']) for r in rows),
                    'trajectories_with_any_recovery': sum(r['recovered_after_first_error'] == 'True' for r in rows),
                    'trajectories_with_recovery_opportunity': sum(r['recovery_eligible'] == 'True' for r in rows)})
    with (OUT / 'transition_summary.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(transitions[0]))
        writer.writeheader(); writer.writerows(transitions)

    cohort_rows = []
    for m in MODELS:
        a = status['cohort_audit'][m]
        cohort_rows.append([NAMES[m], 2155, 2883, 1608, len(eligible[m]),
                            1608 - len(eligible[m]), a['scale0_prediction_mismatch_count']])
    contrast_rows = [[NAMES[m]] + [f'{point[m + "/" + metric] * 100:.2f}' for metric in
                      ('delta_F_publisher', 'delta_F_common', 'delta_T_common', 'D_C', 'D_E')]
                     for m in MODELS]
    ci_rows = [[NAMES[r['model']], r['metric'], f"{r['estimate_pp']:+.2f}",
                f"[{r['source_joint95_low_pp']:+.2f}, {r['source_joint95_high_pp']:+.2f}]",
                f"[{r['cluster_joint95_low_pp']:+.2f}, {r['cluster_joint95_high_pp']:+.2f}]"]
               for r in summary_rows]
    recovery = [r for r in read_csv('recovery_by_first_error')
                if r['cohort'] == 'common_first_observed_correct']
    recovery_rows = []
    for m in MODELS:
        for s in SHIFTS:
            selected = {(float(r['first_error_scale'])): r for r in recovery if r['model'] == m and r['shift'] == s}
            cells = []
            for scale in (.5, 1, 1.5, 2):
                r = selected[scale]
                cells.append(f"{r['recovered']}/{r['first_errors']} ({100 * float(r['recovery_rate']):.1f}%)")
            cells.append(f"NA；首错 {selected[2.5]['first_errors']} 条")
            recovery_rows.append([NAMES[m], s, *cells])
    accuracy = [r for r in read_csv('accuracy') if r['cohort'] == 'common_first_observed_correct']
    accuracy_rows = [[NAMES[m], s] + [f'{100 * float(r["accuracy"]):.2f}' for r in accuracy
                                     if r['model'] == m and r['shift'] == s]
                     for m in MODELS for s in SHIFTS]
    support = [r for r in read_csv('cohort_support') if r['shift'] == SHIFTS[0]
               and r['cohort'] == 'common_first_observed_correct']
    support_rows = [[NAMES[r['model']], r['empty_fixed_classes']] for r in support]

    report = f"""# CNS-Bench 首轮重分析：真实发布预测、仅键匹配

**本轮不升级研究立项，也不追加训练/GPU。严格同源检验的结论仍未定。** 两模型在当前 `(class, seed)` 键匹配近似下，cartoon 在 1.5 附近的首错集中现象仍在；换为最终观察到的全错后缀起点后，预先规定的差分改变量很小。身份与类别支持不足阻止正式的同源稳定性结论，不能把本轮写成否定性实验证明或新贡献。

完成时间：{status['completed_utc']}。原 protocol 于 2026-10-30 10:34:37 UTC 锁定，端点、模型、shift、bootstrap 与 5 pp 资源门槛均未改。证据类型：`EMPIRICAL_KEY_MATCHED_EXPLORATORY`。

## 数据与验收

- 官方 [Edmond V1 results.zip](https://edmond.mpg.de/file.xhtml?fileId=318928&version=1.0) 经正常浏览器下载一次，76,383,186 字节，603 个 ZIP 项目，全部 CRC 通过。只读取四份预先指定 CSV，无图像/权重下载。
- SHA-256：`074a436da05b5af8f35bdf8c7a11e670fe1a13776b320f5ac3ada524a41e80ac`；本地 MD5：`7f88096bcd433c36b91d6b3baeeb752c`。官方页面只留存了缩略 `7f8…52c`，故不能声称完整发布方 MD5 已独立比对。
- CC0 1.0 见官方数据页和固定提交的标注 `info.license`；数据版本 1.0。官方代码 MIT 不替代数据许可。
- 每模型 cartoon 12,930 行、snow 17,298 行；30,228 个唯一键，和标注一一对应；六尺度完整，无缺点轨迹排除。两模型按键合并，缺失键为 0。
- CSV 文件名为 `.png`、标注为 `.jpg`。严格验收先报错；在看到效应前保存兼容说明，仅显式接受相同类别目录与文件名主体的扩展名对应，不修改原数据、不证明图像字节相等。32 项合成测试通过，与真实结果分开保存。

## 样本与身份边界

{table(['模型', '发布 cartoon', '发布 snow', '共同完整键', '两侧首点均正确', '首点资格排除', 'scale0 预测不一致'], cohort_rows)}

共同完整键涉及 91 类；发布样本两侧共同类别集合 K 固定为 93 类。每侧分别有 547 / 1,275 条轨迹没有共同键。跨模型同时满足两侧首点正确的键为 {len(cross)}。各原始样本群和共同完整样本群中的首点错误状态均保留在敏感性表中，不当作删失。

**0 / 1,608 对基图有真实字节或可靠同源标识核验。** scale0 预测不一致分别为 31 和 19 对；一致预测也不能证明同源，不一致本身也不能直接证明图像不同。现阶段只能称“首个已观察尺度正确”，不能认定为 clean baseline。下面的联合 bootstrap 也仅条件于未经身份核验的键对应，不能作为严格源图配对推断。

## 固定主要差分

F 为首次错误位置；T 为最后一段连续错误的起点，且只相对于这六个已观察点，终点正确则 T=∞。T=1.5 不必同时满足 F=1.5。Δ 为 cartoon−snow 的 `Pr(endpoint=1.5)`，所有未出现该事件的合格轨迹仍在分母。D_C=ΔF_common−ΔF_publisher；D_E=ΔT_common−ΔF_common；两者不是因果中介分解。单位均为轨迹比例的百分点，不是准确率变化。

{table(['模型', 'ΔF 发布', 'ΔF 键匹配资格群', 'ΔT 同群', 'D_C', 'D_E'], contrast_rows)}

{table(['模型', '改变量', '估计 pp', '类别分层整键 95% 联合区间', '类别 cluster 95% 联合区间'], ci_rows)}

每套 4,000 次 bootstrap；主方案按类别分层，在相关 cohort 的键并集上联合抽样，保留 shift/scale/model/资格关系。区间同时覆盖两模型的 D_C/D_E 四坐标，未把两组独立区间相减。类别 cluster 对应不同推断目标，仅作敏感性。

上述四个数值区间在两种方案下都落于预定 ±5 pp 内，**但身份门未通过，正式 gate 仍是 `undetermined_source_identity_not_verified`**。不能据此宣布已证实严格同源稳定性、没有机制或没有效应。当前结果没有提供足以扩大投入的预定主差分信号。

原 CSV 每条轨迹均已按 scale 排序，官方“每组首个错误行”与显式最小错误 scale 在四文件中差异为 0。原发布 F 分布、始终正确分母和独立核查见 `original-definition-reproduction-check.json` 与 `event_proportions.csv`；不是重新运行模型的复现。

## 类别等权与跨模型敏感性

固定 K=93 类时，发布样本的 ΔF 类别等权值：ResNet-18 为 21.94 pp、ViT-B 为 15.36 pp；相对原发布轨迹权重分别改变 +5.05 / +1.84 pp。不能隐藏这个权重敏感性，也不能把它事后替换成主要门槛端点。

共同资格群的空类别如下，故其类别等权差分及 D_C/D_E **不估计**，没有删除空类别换成另一个 K：

{table(['模型', '固定 K 中为空的类别 ID'], support_rows)}

类别等权发布估计的 bootstrap 支持不足，未提供该区间；不把有效重采样子集当完整推断。跨模型共同资格群中，ResNet-18 的 ΔF / ΔT / D_E 为 17.95 / 18.10 / +0.14 pp；ViT-B 为 16.22 / 15.50 / −0.72 pp。两模型分别报告，不把模型当独立随机重复。

## 错误后恢复与逐尺度准确率

以下恢复定义为 F 之后任一更晚观察点正确，严格按 F 分层；分母是该首错层的轨迹数，较晚首错可恢复机会较少，比较仅为描述性。

{table(['模型', 'shift', 'F=.5', 'F=1', 'F=1.5', 'F=2', 'F=2.5'], recovery_rows)}

F=0 在此首点资格群没有事件，NA；F=2.5 无后续恢复机会，不能写成 0% 恢复。部分层少于 20 条，比例不稳。Snow 恢复比例可较高，但这不自动构成新机制，也不能替代预先规定的整体差分。T 不是“永久/不可逆失败”。

{table(['模型', 'shift', '0', '.5', '1', '1.5', '2', '2.5'], accuracy_rows)}

准确率表单位为 %；首点 100% 来自资格定义。逐轨迹首错、后缀起点、correct→wrong / wrong→correct 次数、恢复、终点正确保存在 `trajectory_features.csv`；汇总见 `transition_summary.csv`。这些翻转统计不是新概念，ImageNet-P 等已有相关研究。

## 本轮决定与限制

暂停此候选的进一步投入，不训练、不扩模型或 shift，不据此提出论文贡献。若之后继续，缺少的是公开源图身份对应证据和图片格式/发布版本对应证据；应先核对官方可提供的小子集或标识，不自动下载 2.5 GB 整图包。当前正常页面未验得可单取全部所需基图的正式路径，因此没有自行获取图片。

推断只适用于已过滤发布 cohort；匹配和首点正确筛选有选择偏差；不同 shift 的数值尺度不是同等物理强度。基图身份和固定类别支持仍不完整，因而最终状态是“描述性现象保持、严格配对问题未定”，不是负结果关闭。零事件/退化区间也不能证明事件不存在。本轮没有模型训练、推理、GPU 分配、Colab 额度消费、commit 或 push。

图：`failure_distributions.png`、`accuracy.png`、`contrast_changes.png`（另有 SVG）。第三图误差线为类别分层整键 bootstrap 的四坐标 95% 联合区间，灰色为 ±5 pp 资源判断区间。所有图均明确写出源图身份未验证。

来源：[官方论文](https://arxiv.org/html/2507.17651v1)、[官方代码固定提交](https://github.com/odunkel/CNS-Bench/tree/3c09a993007b5844ccd799f8c523a00985aa31ad)、[Edmond 数据集 V1](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP)。
"""
    write_new(OUT / 'RESULTS_ZH.md', report)
    brief = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed_first_pass',
             'evidence_type': status['evidence_type'], 'formal_resource_gate': status['resource_gate'],
             'recommendation': 'Do not escalate research investment; preserve unresolved identity rather than claim a negative primary result.',
             'prediction_download_complete': True, 'prediction_sha256': '074a436da05b5af8f35bdf8c7a11e670fe1a13776b320f5ac3ada524a41e80ac',
             'synthetic_tests_passed': 32, 'bootstrap_replicates_each': 4000,
             'strict_source_identity_verified': False, 'primary_changes': summary_rows,
             'remaining_limits': ['No verified baseline byte/source identity', 'PNG prediction names versus JPG annotation names',
                                  'Fixed class support insufficient in eligible common cohorts', 'Full publisher MD5 not independently read'],
             'training': False, 'inference': False, 'GPU': False}
    write_new(ROOT / 'latest-first-pass-summary.json', json.dumps(brief, ensure_ascii=False, indent=2) + '\n')
    write_new(ROOT / 'LATEST_STATUS_ZH.md', '# 最新状态：真实预测首轮分析完成\n\n'
              '2026-10-30 11:21 UTC 已完成固定 2 shift × 2 model 的 CPU 重分析。正常浏览器下载成功，不再受预测包获取阻塞。\n\n'
              '结论为按键匹配的探索性现象保持；严格同源门未通过，不升级研究立项、不追加训练/GPU，也不将其称为负结果。\n\n'
              '详细结果：[RESULTS_ZH.md](empirical-keymatched-001/RESULTS_ZH.md)；机器回执：[latest-first-pass-summary.json](latest-first-pass-summary.json)。\n\n'
              '初始 README/STATUS/执行摘要保留为当时下载受阻的历史检查点；以本文件和真实结果目录为最新状态。合成数据仅位于 synthetic-validation-001，真实数据结果位于 empirical-keymatched-001。\n')
    print(json.dumps({'report': str(OUT / 'RESULTS_ZH.md'), 'summary': str(ROOT / 'latest-first-pass-summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
