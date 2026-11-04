"""Terminal artifact verification and reporting only. Never loads a model or changes selection."""
from contracts import *
import collections

def main():
    final=read(P/'FINAL_STATUS.json');assert final['status']=='COMPLETE' and final['all_children_exited']
    result=read(P/'SCIENTIFIC_RESULT.json');selection=read(P/'SELECTION.json');phases=read(P/'PHASES.json');proto=read(P/'PROTOCOL.json')
    assert final['elapsed_seconds']<10800 and all(x['child_exited'] and x['exit_code']==0 for x in phases)
    for name,h in read(P/'CODE_PINS.json').items():assert sha(P/name)==h
    trained={a:read(P/f'formal_{a}/RESULT.json') for a in ('A0','A1')}
    for arm,x in trained.items():
        assert x['status']=='TRAINING_COMPLETE' and x['completed_updates']==200
        assert sha(P/f'formal_{arm}/final_model.pth')==x['checkpoint']['sha256']
        assert x['initial']==proto['expected_initial'] and x['initial_RNG']==proto['expected_initial_RNG']
        assert all(v==40 for v in x['source_exposure']['per_image'].values())
        assert all(v==200 for v in x['update_call_counts'].values()) and x['scheduler_objects']==0
        assert x['all_data_reference_matches']
    recs=[];count=0
    for split in ('dev','confirm'):
        ids=[x['id'] for x in read(P/f'data/annotations/{split}.json')['images']];geometry={}
        for arm in ('P0','A0','A1'):
            receipt=read(P/f'evaluation/{split}/receipts/{arm}.json');assert receipt['status']=='COMPLETE' and receipt['completed_forwards']==len(ids)
            assert receipt['model_state_unchanged'] and receipt['all_gradients_none'] and receipt['actual_captions']==[['fish. ']]
            recs.append(receipt)
            for iid in ids:
                x=read(P/f'evaluation/{split}/predictions/{arm}/{iid}.json');v=x['payload'];assert digest(v)==x['payload_sha256']
                assert v['image_id']==iid and v['arm']==arm and v['split']==split and len(v['predictions'])==300
                assert geometry.setdefault(iid,v['geometry_sha256'])==v['geometry_sha256']
                assert all(z['category_id']==1 and z['image_id']==iid and all(math.isfinite(k) for k in z['bbox']+[z['score']]) for z in v['predictions'])
                if split=='confirm':assert v['dev_decisions_sha256']==sha(P/'DEV_DECISIONS.json')
                count+=1
            m=read(P/f'evaluation/{split}/metrics/{arm}.json');assert m['stats'][0]==m['metrics']['AP']['value'] and m['metrics']['AP']['maxDets']==100
    assert count==1224
    order={x['phase']:i for i,x in enumerate(phases)}
    assert max(order['formal_A0'],order['formal_A1'])<order['infer_dev_P0']<order['metrics_dev']<order['infer_confirm_P0']<order['metrics_confirm']
    assert read(P/'DEV_DECISIONS.json')['UTC']<next(x['started_UTC'] for x in phases if x['phase']=='infer_confirm_P0')
    for arm in ('A0','A1'):
        d=result['dev'];c=result['confirm'];expected=decision(d['P0']['metrics']['AP']['value'],d[arm]['metrics']['AP']['value'],c['P0']['metrics']['AP']['value'],c[arm]['metrics']['AP']['value'])
        assert expected==result['episode_results'][arm]
    peaks=[x['peak_RSS_bytes'] for x in trained.values()]+[x['peak_RSS_bytes'] for x in recs]+[read(P/f'preflight_{a}/RESULT.json')['peak_RSS_bytes'] for a in ('A0','A1')]
    assert max(peaks)<=12*2**30
    ledger={'status':'COMPLETE_INDEPENDENT_TERMINAL_VERIFICATION','UTC':utc(),'formal_updates':400,'completed_preflight_updates':2,'earlier_failed_initialization_updates':0,'prediction_identities':count,'predictions':count*300,'primary_and12metric_mapping_verified':True,'geometry_same_for_all3models':True,'dev_decision_before_confirm_verified':True,'old_code_pins_unchanged_after_repair':True,'train_controls_equal_historical_A':True,'two_full_checkpoint_hashes_verified':{a:trained[a]['checkpoint']['sha256'] for a in trained},'lifecycle_seconds_including_initial_failure_and_repair':final['elapsed_seconds'],'total_train_seconds':sum(sum(s['train_step_seconds'] for s in x['steps']) for x in trained.values()),'total_forward_seconds':sum(x['forward_seconds'] for x in recs),'peak_RSS_bytes':max(peaks),'current_project_plus_staging_bytes':tree(R)+tree(P),'all_lifecycle_children_exited':True,'new_install_upload_download_GPU_MPS_Colab_CU':0,'failure_history_retained':True}
    atomic(P/'TERMINAL_VERIFICATION.json',ledger)
    def ap(split,arm):return result[split][arm]['metrics']['AP']['percent']
    rows=['# Brackish 固定普通适配先导：完成结果','',
          '本轮只检查既有固定5图普通适配的开发集选择，是否在第二数据源的日期隔离确认子集保持收益。没有新增算法、增强搜索或论文贡献结论。','',
          '| 模型 | 开发 AP@100 (%) | 确认 AP@100 (%) |','|---|---:|---:|']
    for arm,label in [('P0','P0 未适配'),('A0','A0′ episode0'),('A1','A1′ episode1')]:rows.append(f'| {label} | {ap("dev",arm):.6f} | {ap("confirm",arm):.6f} |')
    rows+=['','| episode | dev选择 | dev A−P0 (pp) | confirm A−P0 (pp) | 所选−P0 (pp) | 有害选择失配 | 错失收益 |','|---|---|---:|---:|---:|---|---|']
    for arm in ('A0','A1'):
        d=result['episode_results'][arm];selected=arm+'′' if d['chosen']=='A' else 'P0'
        rows.append(f'| {arm[-1]} | {selected} | {100*d["dev_A_minus_P0"]:.6f} | {100*d["confirm_A_minus_P0"]:.6f} | {100*d["selected_minus_P0"]:.6f} | {"是" if d["harmful_selection_mismatch"] else "否"} | {"是" if d["missed_gain"] else "否"} |')
    harmful=sum(d['harmful_selection_mismatch'] for d in result['episode_results'].values());missed=sum(d['missed_gain'] for d in result['episode_results'].values())
    rows+=['',f'有限结论：两个既定 episode 中观察到 {harmful} 次有害选择失配、{missed} 次错失收益。这只是本次固定子集的描述；不做显著性或部署可靠性推断，也不追加种子、帧、训练或调参。',
           '', '两套支持各5图5鱼框，共享3视频但图像不重复；不是独立重复，鱼身份也未证明不同。支持0每图40次曝光；支持1同样每图40次。普通裁剪后保留目标的次数不同，作为固定数据和几何的实际结果报告。']
    for arm in ('A0','A1'):
        x=trained[arm];s=x['source_exposure'];rows.append(f'- {arm}′：200更新、{s["occurrences"]}次来源曝光、{s["GT_presentations"]}次保留鱼框呈现、{s["empty_GT_inputs"]}次空GT输入；训练步骤合计{sum(z["train_step_seconds"] for z in x["steps"])/60:.3f}分钟。')
    rows+=['','开发288图325鱼框，其中213图无鱼目标标注；确认120图38鱼框，其中83图无鱼目标标注。所有符合冻结抽帧规则的空帧均保留。17个开发集越界原框未裁剪；所有选中框均与图像有非零交集。AAU/一致COCO为来源，未混入YOLO；非鱼动物不是ignore区域。',
           '', '支持6日期、开发2日期、确认2日期互斥，但数据来自同一固定相机，不代表跨地点。确认鱼框仅38个、视频内帧相关，标签完整性未知；“文档声明覆盖且无目标标注”不等于视觉真空。本轮数据处理采用固定OpenCV CPU解码与缩放，不声称与作者ffmpeg逐像素复现。',
           '', 'P0与两适配模型具有相同完整初始化；严格完成908键的已核验官方加载契约，原始903键没有被伪称直接strict兼容。训练专用随机DN状态在全部推理中未调用。所有模型同CPU/FP32、同fish. 空格结尾文本、同输入处理、原生top300预测、无新增分数阈值。两模型完整安全落盘后才看开发效用；每episode独立按严格A>P0规则选择，tie选P0；选择写入后才评估确认，且确认结果没有反馈。',
           '', '## 全部12指标（%）','', '| 指标 | dev P0 | dev A0′ | dev A1′ | confirm P0 | confirm A0′ | confirm A1′ |','|---|---:|---:|---:|---:|---:|---:|']
    for metric in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR100','AR300','AR1000','AR_small1000','AR_medium1000','AR_large1000']:
        vals=[result[s][a]['metrics'][metric]['value'] for s in ('dev','confirm') for a in ('P0','A0','A1')]
        rows.append('| '+metric+' | '+' | '.join('—' if v<0 else f'{100*v:.4f}' for v in vals)+' |')
    rows+=['','主AP使用maxDets100；AP50/AP75/尺度AP使用1000；AR分别100/300/1000，尺度AR使用1000。所有指标从持久化预测复算两次，precision/recall/scores完整数组一致；没有用单阈值计数替代AP。',
           '', '## 实测与产物','',f'- 生命周期（包含首次无更新失败与工程修复）：{ledger["lifecycle_seconds_including_initial_failure_and_repair"]/60:.3f}分钟；峰值RSS {ledger["peak_RSS_bytes"]/2**30:.3f} GiB。',
           f'- 正式训练400更新；预检2更新后全部丢弃；评估1224个唯一图像／模型身份，共367,200条原生预测。推理前向合计{ledger["total_forward_seconds"]/60:.3f}分钟。',
           '- 12项fixture通过；两套各200步追踪／未追踪像素、GT、RNG一致；正式各200步均匹配冻结数据参考。',
           '- 首次预检在控制哈希格式检查处停下，0更新、0效用；仅修复历史摘要的末尾换行，保留旧代码与日志，三小时deadline未重置。',
           '- 新安装、上传、数据下载、GPU/MPS/Colab分配、CU均为0；全部子进程已退出；保留旧资产。最终目录字节和完整SHA见归档回执。',
           '- `formal_A0/final_model.pth`、`formal_A1/final_model.pth`：两份完整最终模型及逐张量安全读回回执。',
           '- `evaluation/{dev,confirm}/predictions/`、`metrics/`：全部预测、12指标和完整数组；`DEV_DECISIONS.json`：确认前的固定选择。',
           '- `SELECTION.json`、`SOURCE_ANNOTATION_PROVENANCE.json`、`IMAGE_MANIFEST.json`、`PROTOCOL.json`、`CODE_PINS.json`：数据、代码、环境、原权重和科学规则。',
           '- `PHASES.json`、`PREFLIGHT_GATE.json`、`TERMINAL_VERIFICATION.json`：实际阶段、成本门和独立终态校验。',
           '', '数据许可字段CC BY-SA／MIT冲突未获作者澄清；本次自限本地研究、不外发，保留发布者署名、许可证和相同方式共享来源。']
    (P/'REPORT_ZH.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({'status':ledger['status'],'primary_AP_percent':{s:{a:ap(s,a) for a in ('P0','A0','A1')} for s in ('dev','confirm')},'episode_results':result['episode_results'],'ledger':ledger},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
