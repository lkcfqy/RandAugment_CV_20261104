"""Render all frozen comparisons after verified completion; no model code imported."""
import csv,datetime,json,pathlib
P=pathlib.Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
status=read('FINAL_STATUS.json');assert status['status']=='COMPLETE' and status['all_children_exited']
matrix=read('ALL_696_COMPLETE.json');assert matrix['records_count']==696 and matrix['images']==232
verified=read('METRIC_RECOMPUTE_VERIFIED.json');assert verified['conditions']==96
comp=read('COMPARISON.json');selection=read('SELECTION.json');strata=read('STRATA.json')
phases=read('PHASES.json');frozen=read('FROZEN.json')
main=comp['metrics']['pooled'];delta=comp['contrasts']['pooled']
conclusion='两份固定5图/U200适配模型在此次前缀限帧子集上均保持正收益，构成有限确认。' if all(delta[k]>0 for k in ['A0_minus_P0_pp','A1_minus_P0_pp']) else '此次前缀限帧子集出现反转或分歧，适配收益的可保持性受到限制。'
def fmt(metric):return 'NA' if metric['value']<0 else f"{metric['percent']:.4f}"
rows=[]
for group,models in comp['metrics'].items():
    for arm,metrics in models.items():
        out={'group':group,'model':arm}
        out.update({k:v['percent'] if v['value']>=0 else '' for k,v in metrics.items()});rows.append(out)
with (P/'ALL_METRICS.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
receipts=[read('receipts/model_20261031_'+a+'.json') for a in ['P0','A0','A1']]
peak=max(x['peak_RSS_bytes'] for x in receipts)
lines=['# 固定前缀限帧确认：P0 / A0 / A1','',conclusion,'',
'本轮为前瞻性抽样修订后的单次评估。原2014图全池计划保留；因相关文件序列冗余及约5小时成本，先于任何确认效用决定保留全部29前缀，每组固定取8图。各组按文件名末尾帧号、imageID、文件名排序，索引floor((j+.5)n/m)，m=min(8,n)。不按GT、外观、难度或模型分数筛选。帧号是文件序列信息，非已核实的时间戳。','',
'最终清单232图576框，三个模型各232次，合计696个唯一持久化预测。全部模型完成后才统一评价并从持久化记录复算。没有新训练、强度/prompt/阈值/maxDets更改；未评估官方test或其余1782张未选确认图。','',
'## 主结果','',
'主问题仅为固定5图/U200适配相对未适配P0的收益是否保持。以下为此次固定子集pooled指标，不能当作2014图全池逐帧AP。','',
'|模型|AP@100 (%)|AP50@1000 (%)|AP75@1000 (%)|AR100 (%)|相对P0 AP差(pp)|',
'|---|---:|---:|---:|---:|---:|']
for a in ['P0','A0','A1']:
    d=0 if a=='P0' else delta[a+'_minus_P0_pp']
    lines.append('|'+a+'|'+'|'.join(fmt(main[a][k]) for k in ['AP','AP50','AP75','AR100'])+f'|{d:+.4f}|')
lines += ['',conclusion+' 不声称新算法、新机制或统计显著性；这两个A来自两个已经固定的支持集、同一seed。无论方向如何，不补抽帧、不自动全池或追加训练。','',
'## 预先登记的数字habitat分层（次要描述）','',
'“共享/不共享”只指是否与已见组共享首段数字selector；不是已验证的新站点、视频或独立habitat。两个层全部报告，不能用较有利层代替总体。','',
'|层|前缀|图|框|P0 AP@100|A0 AP@100|A1 AP@100|A0−P0(pp)|A1−P0(pp)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in ['numeric_habitat_shared','numeric_habitat_unshared']:
    size=strata['sizes'][g];ms=comp['metrics'][g];ds=comp['contrasts'][g]
    lines.append(f"|{g}|{size['prefixes']}|{size['images']}|{size['boxes']}|"+'|'.join(fmt(ms[a]['AP']) for a in ['P0','A0','A1'])+f"|{ds['A0_minus_P0_pp']:+.4f}|{ds['A1_minus_P0_pp']:+.4f}|")
lines += ['','## 全部29前缀描述','',
'每组8图，统一原标注。前缀不是已证实独立场景；均匀抽帧不使相关帧变成独立样本。此表不用于事后挑组或显著性推断。','',
'|前缀|框|P0 AP@100|A0 AP@100|A1 AP@100|A0−P0(pp)|A1−P0(pp)|','|---|---:|---:|---:|---:|---:|---:|']
gt=read('data/annotations.json')['annotations']
for prefix,ids in sorted(strata['groups'].items()):
    g='prefix_'+prefix;ms=comp['metrics'][g];ds=comp['contrasts'][g];boxes=sum(a['image_id'] in ids for a in gt)
    lines.append(f'|{prefix}|{boxes}|'+'|'.join(fmt(ms[a]['AP']) for a in ['P0','A0','A1'])+f"|{ds['A0_minus_P0_pp']:+.4f}|{ds['A1_minus_P0_pp']:+.4f}|")
lines += ['','## 实现、检查与资源','',
'全部使用同一官方MMDetection3.3/MMCV2.1/PyTorch2.1 CPU实现、FP32、torch4线程/interop1/OpenCV1。BGR uint8解码→Color1恒等→原FixScaleResize→原LoadAnnotations，实际caption固定为 `fish. `。原生top300预测，主AP maxDets100；AP50/AP75及尺度AP为1000，AR分别100/300/1000，尺度AR1000。全部12项及完整precision/recall/scores数组保留于96份总体/分层/分组条件结果；无GT的尺度项在CSV留空，不能把官方-1当真实负AP。','',
'P0保持A0/A1的同一完整初始状态：900个原权重活动张量、7个固定log_scale及仅训练用的seed固定DN状态。不是903键原文件直接strict加载，DN在本次推理未调用。A0/A1严格加载既有完整908张量checkpoint；模型全部weights_only安全读取，状态未改变，无梯度、无大权重复制。','',
'8项无模型测试通过，确认抽样索引、完整标签、696身份集合、原加载函数AST、原前向与后处理片段、指标映射及原子落盘/冲突检测。准备时系统Python与实验Python单调时钟起点不同，安全门在0图/0模型时停止；改由同一实验Python记录时钟，保留原UTC截止，修订与失败日志已归档。','',
'|阶段|秒|','|---|---:|']
for phase in phases:lines.append(f"|{phase['label']}|{phase['elapsed_seconds']:.3f}|")
lines += ['',f"生命周期实际{status['lifecycle_seconds']/60:.3f}分钟；含本轮准备的总墙钟{status['total_workflow_seconds']/60:.3f}分钟，上限75分钟。三个推理进程最高RSS {peak/1024**3:.3f}GiB，上限12GiB。三个模型完整预测及全部96条件指标/数组已精确复算，所有子进程已退出。新增训练、安装、上传、GPU、MPS、Colab、CU均0。",'',
'独立性限制不因正结果消失：29前缀与两support/dev/已知QA前缀隔离，跨前缀视频关系未知；相关帧不视为独立样本。历史3张test曾做图像QA，本轮test没有模型效果或选择用途。该计划在已有dev探索后冻结，不能包装成从研究起点预注册的独立场景确认。','',
f"冻结清单SHA256：`{frozen['file_sha256']['SELECTION.json']}`。执行协议SHA256：`{frozen['run_protocol_sha256']}`。全部数据、模型、代码、环境身份见FROZEN/RUN_PROTOCOL和最终manifest。",'']
(P/'REPORT_ZH.md').write_text('\n'.join(lines))
decision={'status':'COMPLETE','primary_dataset':'232 fixed prefix-capped train images,576GT,29prefixes','interpretation':conclusion,'primary_differences_pp':delta,'do_not_generalize_to_full2014_or_independent_scenes':True,'next_training_or_extra_evaluation_authorized':False,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'SCIENTIFIC_DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'decision':decision,'primary_AP_percent':{a:main[a]['AP']['percent'] for a in main},'peak_RSS_GiB':peak/1024**3,'total_workflow_minutes':status['total_workflow_seconds']/60},ensure_ascii=False))
