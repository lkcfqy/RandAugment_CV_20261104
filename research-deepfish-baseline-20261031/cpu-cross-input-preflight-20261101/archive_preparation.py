"""Archive completed CPU engineering evidence; no model imports or predictions."""
import datetime
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import shutil
from urllib.parse import urlparse

P = pathlib.Path(__file__).resolve().parent
R = P.parent

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()

def save(name, obj):
    dest=P/name
    assert not dest.exists(), 'Refuse archive overwrite'
    dest.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def size():
    return sum(f.stat().st_size for f in R.rglob('*') if f.is_file())

protocol=json.loads((P/'PROTOCOL.json').read_text())
assert sha(P/'PROTOCOL.json')=='0b4101c0e9e41904c8dc23618439fa2499815eb136ea106bd3c48ef1ed02e305'
terminal=json.loads((P/'receipts/r1/CPU_PROBE_TERMINAL.json').read_text())
assert terminal['status']=='ENGINEERING_PREFLIGHT_COMPLETE_NOT_FULL_DIAGNOSIS'
models=[]
for spec in protocol['models']:
    label=str(spec['seed'])+'_'+spec['arm']
    rec=json.loads((P/'receipts/r1'/('probe_'+label+'.json')).read_text())
    assert rec['status']=='ENGINEERING_PASS'
    assert rec['forward_calls_started']==rec['forward_calls_completed']==2
    assert rec['load']['checkpoint_sha256']==spec['sha256']
    assert rec['load']['strict'] and rec['load']['state_tensor_count']==908
    assert rec['load']['all_tensors_byte_identical'] and rec['load']['all_checkpoint_values_finite']
    assert not rec['load']['missing_keys'] and not rec['load']['unexpected_keys']
    assert all(c['output_finite'] and c['output_device']=='cpu' for c in rec['conditions'])
    assert rec['dev_test_images']==0 and not rec['metrics_computed'] and not rec['predictions_saved']
    models.append(rec)
first=json.loads((P/'receipts/probe_20261031_A.json').read_text())
assert first['forward_calls_started']==1 and first['forward_calls_completed']==0
assert first['reason']=="AttributeError: 'dict' object has no attribute 'num_heads'"
threads=json.loads((P/'receipts/THREAD_CONTRACT_CHECK.json').read_text())
assert threads['prepared_tensors_gt_metadata_byte_identical'] and threads['model_instances']==0
for k in ['support_annotation','dev_annotation']:
    assert sha(protocol[k]['path'])==protocol[k]['sha256']
dev=json.loads(pathlib.Path(protocol['dev_annotation']['path']).read_text())
assert len(dev['images'])==64 and len(dev['annotations'])==187
assert {a['id'] for a in dev['annotations']} >= {4195,4660}

packages={}
for report_name in ['build_tools_install.json','core_install.json']:
    report=json.loads((P/'receipts'/report_name).read_text())
    for item in report['install']:
        name=item['metadata']['name']; version=item['metadata']['version']
        download=item['download_info']; url=download['url']; parsed=urlparse(url)
        if parsed.scheme=='file':
            assert name.lower()=='mmcv'
            assert download['archive_info']['hashes']['sha256']=='6bc8018e724dc53dbea36fe4db5777da25cd0b67a912b2e802d2cba7b4ce3894'
        else:
            assert parsed.hostname in ['files.pythonhosted.org','pypi.org','download.openmmlab.com'],url
        packages[name]={'version':version,'download_info':download,'installed_version_verified':True}
for name,item in packages.items():
    assert md.version(name)==item['version'], (name, md.version(name), item['version'])
lock={'status':'INSTALLED_OFFICIAL_DISTRIBUTIONS_WITH_DOCUMENTED_WHEEL_METADATA_DEFECTS',
      'packages':packages,'pip_freeze_file':'logs/pip_freeze.log',
      'pip_check_exit_code':1,'pip_check_diagnosis':'receipts/WHEEL_METADATA_DIAGNOSIS.json',
      'dependency_missing_or_version_conflicts':False,'native_import_gate':'receipts/OFFICIAL_IMPORT_GATE.json',
      'native_source_compilation':False,'fairscale_CUDA_extensions':False,'metadata_modified':False,
      'install_history_note':'Build-tool installation temporarily brought packaging26.3; the subsequent frozen core requirements installed packaging24.2. Final versions are validated against the last applicable installation receipt. No environment change was made during archive bookkeeping.',
      'interpreter_receipt':'receipts/python_source.json','system_environment_modified':False}
save('ENVIRONMENT_LOCK.json',lock)

forward_max=max(c['forward_seconds'] for m in models for c in m['conditions'])
pipeline_max=max(c['pipeline_seconds'] for c in threads['records'] if c['requested_cv2_threads']==0)
composed=max(terminal['slowest_probe_pipeline_plus_forward_seconds'],forward_max+pipeline_max)
component_seconds=1.5*terminal['model_load_seconds_total']+1024*composed+600
old_files=[]
for base in [R/'recovery-v5-20261031/gpu-run/artifacts',R/'repeat-seed20261101-budget25-20261101/gpu-run/artifacts']:
    for f in sorted(base.glob('evaluate_*_predictions.bbox.json')):
        old_files.append({'path':str(f.relative_to(R)),'bytes':f.stat().st_size})
assert len(old_files)==4
prediction_plan_bytes=8*max(x['bytes'] for x in old_files)
before=size(); cap=6*1024**3
cost={'frozen_formula_seconds':terminal['planning_512_seconds'],
      'frozen_formula':protocol['cost_rule_frozen']['seconds_formula'],
      'sequential_OpenCV_proposed_runtime_component_estimate_seconds':component_seconds,
      'component_formula':'1.5*sum(4 measured load seconds)+512*2*max(original8 combined timing maximum, maximum measured torch forward + maximum measured sequential pipeline)+600',
      'measured_4_load_seconds':terminal['model_load_seconds_total'],
      'load_reserve_seconds':1.5*terminal['model_load_seconds_total'],
      'measured_forward_max_seconds':forward_max,
      'measured_sequential_pipeline_max_seconds':pipeline_max,
      'forward_reserve_seconds':1024*composed,'persistence_8_evaluations_checks_reserve_seconds':600,
      'measured_full_512_seconds':None,'full_512_forward_calls':0,
      'actual_successful_probe_supervisor_seconds':terminal['actual_supervisor_seconds'],
      'actual_prior_aborted_probe_seconds':terminal['prior_aborted_process_seconds'],
      'peak_process_RSS_bytes':terminal['maximum_peak_RSS_bytes'],
      'RSS_measurement':'macOS resource.getrusage(RUSAGE_SELF).ru_maxrss; no physical RAM query',
      'single_support_image_timing_limit':True,'not_a_guaranteed_wall_time_upper_bound':True,
      'prior_prediction_sizes_stat_only':old_files,'prior_prediction_contents_opened':False,
      '8_output_prediction_JSON_size_extrapolation_bytes':prediction_plan_bytes,
      'proposed_total_new_formal_output_cap_bytes':100*1024**2,
      'research_bytes_before_report':before,'research_cap_bytes':cap,'cap_margin_bytes_before_report':cap-before,
      'filesystem_free_bytes':shutil.disk_usage(R).free,
      'future_output_checkpoint_copies':0,'upload_bytes':0,'new_CU':0,'new_GPU_seconds':0,
      'unknowns':['64-image sustained CPU throughput and thermal/background load','full8-condition evaluation/persistence timings','full-run peak RSS beyond one image','reproducibility on any different CPU/environment'],
      'same_CPU_original_and_transformed_required':True}
save('COST_EVIDENCE.json',cost)

plan={'status':'PROPOSED_PARENT_REVIEW_REQUIRED_NOT_EXECUTED',
      'protocol_sha256':sha(P/'PROTOCOL.json'),'question':protocol['question'],
      'models':[{'seed':x['seed'],'arm':x['arm'],'checkpoint_sha256':x['sha256'],'config_sha256':x['config_sha256']} for x in protocol['models']],
      'inputs':{'dev_images':64,'GT_boxes':187,'annotation_sha256':protocol['dev_annotation']['sha256'],
                'image_order':'unchanged dev64.json image order as loaded by official CocoDataset',
                'conditions_in_order':[1.0,0.8],'forward_calls_total':512,'model_loads':4,
                'official_test':False,'support_in_formal_evaluation':False},
      'runtime':{**protocol['runtime'],'cv2_setNumThreads':0,'required_cv2_getNumThreads':1,
                 'Config_container':'mmengine.Config/ConfigDict as official Runner',
                 'pipeline_threads_revision':'No-model exact packed tensor/GT/metadata equality verified in THREAD_CONTRACT_CHECK; eight model probes were under original GCD setting, not rerun.',
                 'checkpoint_load':'weights_only=True CPU; strict=True; exact all-key/dtype/shape/bytes check; no allowlist/fallback'},
      'metric_contract':protocol['future_formal'],
      'primary_contrast':{'per_seed_input_gap':'Delta(c)=AP(B,c)-AP(A,c)',
                          'interaction':'Delta(0.8)-Delta(1.0)',
                          'units':'percentage points of original primary AP@maxDets100',
                          'direction':'two-sided descriptive result; no threshold/strength selection',
                          'AP50_AP75_scale_AR':'secondary explanation, preserving their original maxDets1000 convention'},
      'execution_sequence':['Revalidate local protocol/assets/package pins, CPU-only flags and research disk margin.',
                            'Use four fresh processes in frozen checkpoint order; each runs all64 original inputs followed by all64 factor0.8 inputs; no optimizer, gradients, warmups or new seed.',
                            'Persist complete top300 predictions per image and local atomic receipts for all8 conditions; no score-based filtering or viewing interim AP.',
                            'After all8 conditions exist, evaluate all on the same original dev annotation and official COCO semantics; preserve raw metric/precision arrays and finite outputs.',
                            'Report both seeds and all8 scores, original primary metric and secondary metrics; never mix CPU scores with old GPU scores.',
                            'Stop on contract/resource/finite-value failure; retain partial execution status without converting it into a complete scientific result.'],
      'disk':{'research_cap_bytes':cap,'planned_new_output_cap_bytes':100*1024**2,'retain_free_research_margin_bytes':80*1024**2,'duplicate_models':False},
      'time':{'planning_seconds':component_seconds,'measured_full_run':False,'planning_note':'About48.13min; retain a roughly1h local window as a practical estimate, not a promise.'},
      'interpretation_limit':protocol['interpretation_limit'],'no_new_training_or_test_or_third_seed':True,
      'entrypoint_status':'No full-evaluation entrypoint has been executed or added. Existing cpu_preflight*.py are bounded support-only probes and refuse repeated execution.'}
save('FULL_DIAGNOSTIC_PLAN.json',plan)

rows=[]
for m in models:
    a,b=m['conditions']
    rows.append(f"| {m['model']} | {m['load']['seconds']:.3f} | {a['pipeline_plus_forward_seconds']:.4f} | {b['pipeline_plus_forward_seconds']:.4f} | {m['peak_RSS_bytes']/1024**3:.4f} |")
report=f'''本机 CPU 准备和有限工程检查完成。已有4个完整模型均通过安全、严格、逐张量加载核对；同一张预先按SHA固定的support图完成8次前向，全部输出有限。完整512次dev诊断未启动，本次没有效用结果。是否开展完整诊断仍待父线程审核。

实际使用项目专用 CPython3.11.14 ARM64 / Torch2.1.0 / torchvision0.16.0 / MMCV2.1.0 / MMDetection3.3.0 / MMEngine0.10.7，NumPy1.24.3、OpenCV4.11.0.86、transformers4.34.0及其余精确版本见 [ENVIRONMENT_LOCK.json](ENVIRONMENT_LOCK.json) 和 [pip freeze](logs/pip_freeze.log)。Python来自此前已允许的官方 Astral 固定发行；核心包仅来自PyPI/OpenMMLab。未修改系统环境、模型/算子源码或安装第三方fork；Fairscale仅构建普通Python wheel，CUDA扩展关闭。已有BERT safetensors在本机复用，未下载新模型权重。

原pip check返回1，不能写作“pip check通过”。诊断确认：Torch官方ARM64文件名与Mach-O/RECORD正确，内嵌WHEEL却写x86_64；Ninja的Tag出现在空行之后，标准header parser读不到。依赖缺失/版本冲突均为0。保留原失败日志和未修改的包元数据；官方实际导入、6项原实现文件pins与随后的完整模型检查通过。证据在 [WHEEL_METADATA_DIAGNOSIS.json](receipts/WHEEL_METADATA_DIAGNOSIS.json) 和 [OFFICIAL_IMPORT_GATE.json](receipts/OFFICIAL_IMPORT_GATE.json)。没有为了通过检查改wheel标签、任意白名单或降级pip。

每个checkpoint共有908个state张量、691,880,148个原始张量字节；均weights_only=True在CPU加载，strict=True，无missing/unexpected，全部键、dtype、shape与字节精确匹配，所有浮点值有限。四个文件hash、完整状态hash及原模型/数据hash详见 [PROTOCOL.json](PROTOCOL.json) 与 [4份回执](receipts/r1)。无需unsafe pickle、动态白名单或trust_remote_code。

测速只用support1892 / 9870_Gerres_f000022.jpg，SHA256=117ea3546c7b67e1fbc9a213cc7cd920b934b0edaea65af28b1d72c076c22071。原始1080×1920 uint8/BGR，经原pipeline得到3×749×1333输入。Color位于cv2解码后、原FixScaleResize前；Color(prob=1,level=0,min=max1或0.8)保持不变。A逐像素恒等，B逐字节匹配固定cv2饱和度混合，两者同样消费1个NumPy随机数，几何、GT和pack后的metadata一致。原始annotation未改。

以下为实测秒数，RSS为各新进程的最高值，未查询物理内存容量。模型eval、FP32、inference_mode，Torch线程4、interop1，没有autocast或compile。

| 模型 | 构建+安全加载+完整校验秒 | 原图管线+前向秒 | 0.8管线+前向秒 | 峰值RSS GiB |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

成功四模型阶段用时{terminal['actual_supervisor_seconds']:.3f}秒。此前CPU入口曾把普通dict传给官方需要ConfigDict属性访问的代码：A安全加载完成后首次前向中止，未返回预测。已保留失败记录，并在无效用结果时以原官方Runner同样的Config包装修正入口；另一次静态字段断言写错路径，在模型启动前已修正。合计5次模型构建/加载、9次前向调用开始、8次完整前向、1次中止；中止阶段另用{terminal['prior_aborted_process_seconds']:.3f}秒，没有隐藏重试。科学配置/权重/图像/颜色强度均未改，见 [INFRA_REVISION_R1.json](INFRA_REVISION_R1.json)。

线程有一项必须明确：八次模型测速的OpenCV是GCD后端，setNumThreads(4)并未形成4线程上限，getNumThreads返回18；这不是实测并发线程数。Torch的4/1线程设置已核实。为满足未来固定线程运行，额外只做了同一support图的无模型管线检查：官方setNumThreads(0)令OpenCV顺序执行且getNumThreads=1；两种颜色的输入张量、GT与metadata均同先前设置逐字节相同。没有再次运行模型。其依据是 [OpenCV4.11官方parallel.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp)，测试见 [THREAD_CONTRACT_CHECK.json](receipts/THREAD_CONTRACT_CHECK.json)。未来建议Torch4/interop1、OpenCV顺序执行，保留其他线程环境变量4。

成本：预先冻结公式给出{terminal['planning_512_seconds']:.3f}秒（{terminal['planning_512_seconds']/60:.3f}分钟）。为纳入OpenCV顺序管线，将实测最慢Torch前向{forward_max:.6f}秒与单独实测最慢顺序管线{pipeline_max:.6f}秒相加，再与原8次最大合并时延取更大值，沿用2倍前向系数、1.5倍4次加载和600秒落盘/8次COCO复算余量，得到{component_seconds:.3f}秒，约{component_seconds/60:.2f}分钟。这是分部实测支持的保守规划数，不是512次实测或硬上界；未测完整持续热负载、其他64图内容变化、完整复算/落盘时间，不能保证一小时内结束。已完成的环境准备与一次入口失败单独记账，不伪装成正式512耗时。详见 [COST_EVIDENCE.json](COST_EVIDENCE.json)。

归档前研究目录{before:,}字节，6GiB上限内余{(cap-before)/1024**2:.2f}MiB，物理磁盘可用{shutil.disk_usage(R).free:,}字节。仅stat旧预测文件大小，未读取其分数；8份新预测JSON按历史最大文件大小外推约{prediction_plan_bytes/1024**2:.2f}MiB。建议新诊断全部产物上限100MiB、保留80MiB研究目录余量，不复制checkpoint。最终磁盘数见归档回执。

正式计划见 [FULL_DIAGNOSTIC_PLAN.json](FULL_DIAGNOSTIC_PLAN.json)：同四checkpoint×同64dev×1.0/0.8，共512前向、4次加载；原图与变换图都在同一CPU环境重算；保留原COCO口径，主AP maxDets100，AP50/AP75等原报告项maxDets1000，完整top300预测，不做阈值筛选。4195/4660原bbox/area和官方先resize后加载标签路径保持。每seed预先比较Delta(c)=AP(B,c)−AP(A,c)及Delta(0.8)−Delta(1.0)，两侧描述，不依据结果改条件或数量。只有全8条件完成后汇总效用，官方test不进入入口。

本轮模型/算子未移植，未训练、未新增seed、未读dev/test预测分数，未上传、未使用GPU/MPS/Colab；新增CU=0。历史账本仍1.8088330815795528CU、5443.899788473129GPU秒，未清零。此前被拒绝的sysctl物理内存读取未重试，也未换路径查询。所有CPU子进程已经正常结束。

无需模型即可确认：B固定0.8只让同一原图每次曝光经过同一颜色映射，没有新增随机颜色选择；标准随机几何仍在。它不代表一般随机增强。未来交叉输入最多回答固定训练颜色与人工输入颜色匹配是否改变A/B差距，不能当真实域偏移、部署预处理、机制证明或可靠科研贡献。

剩余事项是父线程对完整本机512诊断的GO，以及据已审核计划增加小型正式评估入口。当前没有已启动、待重试或执行结果不确定的模型作业；旧support测速入口带一次性保护，不应再次运行。
'''
assert not (P/'REPORT_ZH.md').exists()
(P/'REPORT_ZH.md').write_text(report)
save('PREPARATION_TERMINAL.json',{'status':'COMPLETE_PREPARATION_ONLY_PARENT_REVIEW_PENDING',
     'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'successful_engineering_forwards':8,
     'aborted_engineering_forwards':1,'actual_model_loads_including_aborted_attempt':5,
     'official_wheel_native_import_and_model_path':'PASS','pip_check':'FAILED_METADATA_DEFECTS_DOCUMENTED',
     'new_model_forwards_after_thread_check':0,'full_dev_forwards':0,'metrics_computed':False,
     'CPU_model_processes_finished':True,'new_CU':0,'new_GPU_seconds':0,'GPU_MPS_Colab_actions':0,
     'upload_bytes':0,'system_environment_changed':False,'physical_RAM_read_retried':False,
     'research_bytes_before_manifest':size(),'remaining_blocker':'Parent review/GO for full512; no full-evaluation entrypoint created/executed.',
     'actions_denied_this_preparation':[],'uncertain_executions':[]})

excluded={'.venv','.python','.cache','tmp'}
entries=[]
for current,dirs,files in os.walk(P):
    dirs[:]=[d for d in dirs if d not in excluded]
    for name in sorted(files):
        path=pathlib.Path(current)/name
        assert name not in ['DELIVERABLE_MANIFEST.json','ARCHIVE_RECEIPT.json']
        entries.append({'path':str(path.relative_to(P)),'bytes':path.stat().st_size,'sha256':sha(path)})
entries.sort(key=lambda x:x['path'])
save('DELIVERABLE_MANIFEST.json',{'files':entries,'excluded_environment_directories':sorted(excluded),
     'environment_reproducibility':'ENVIRONMENT_LOCK, pip installation reports, source archive/wheel hashes and installed RECORD checks retained; environment tree itself excluded.',
     'external_assets':'Frozen PROTOCOL plus LOCAL_PROBE_READY and per-model strict-load receipts pin all reused assets.'})
manifest_hash=sha(P/'DELIVERABLE_MANIFEST.json')
save('ARCHIVE_RECEIPT.json',{'status':'ARCHIVED','manifest_sha256':manifest_hash,'manifest_entries':len(entries),
     'listed_file_bytes':sum(x['bytes'] for x in entries),'research_bytes_before_receipt':size(),
     'research_cap_bytes':cap,'cap_margin_bytes_before_receipt':cap-size(),
     'full512_run_started':False,'new_CU':0,'uploads':0})
assert size()<cap-80*1024**2
print(json.dumps({'status':'ARCHIVED_PREPARATION_ONLY','manifest_sha256':manifest_hash,
                  'entries':len(entries),'research_bytes':size(),'planning_minutes':component_seconds/60,
                  'new_CU':0,'uploads':0,'full512_started':False}))
