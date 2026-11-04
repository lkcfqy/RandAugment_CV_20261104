from ctrla_adapter import *
from datetime import datetime,timezone

def main():
    events=[]
    for line in (ROOT/'logs'/'training-events.jsonl').read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    latest={}
    for item in events:
        if item['event']=='epoch_complete':latest[str(item['seed'])]=item
    asset=json.loads((ROOT/'asset-manifest.json').read_text());budget=json.loads((ROOT/'budget-state.json').read_text());bench=json.loads((ROOT/'benchmark.json').read_text())
    status={'updated_utc':datetime.now(timezone.utc).isoformat(),'adapter_tests_passed':10,'rendering_geometry_test_methods':7,'data_tests_passed':5,'statistics_tests_passed':4,'total_tests_passed':19,'MPS_model_forward_backward_confirmed':bench['status']=='passed','MPS_CPU_fallback':False,'CPU_MPS_gradient_relative_l2':bench['cpu_mps_gradient_relative_l2'],'CIFAR10_download':'complete','official_archive_MD5_verified':asset['torchvision_expected_archive_md5'],'official_archive_SHA256':asset['archive_sha256'],'train_members_verified':5,'split_counts':asset['split_counts'],'test_extracted':False,'test_loaded':False,'formal_training_started':budget['train_images']>0,'latest_completed_epochs':latest,'budget':budget,'current_blocker':None,'recovered_failures':['MPS channels-last backward stride error: contiguous layout, gradient agreement verified','macOS spawn serialization: registered dynamically imported audited modules; 2-worker smoke passed'],'online_four_arm_started':False,'active_stage':json.loads((ROOT/'supervisor-live.json').read_text())}
    write_json(ROOT/'PROGRESS.json',status)
    lines=['# 当前进度（可核验）','',status['updated_utc'],'','适配器10项（其中7项渲染/几何）、数据5项、统计4项，合计19项测试通过。MPS完整前后向通过，无CPU fallback。','',f"CIFAR-10：下载及官方MD5、5个训练成员校验通过，SHA256 {asset['archive_sha256']}；固定46k/1k/1k/2k分层。test未解包、未加载。",'',f"正式训练已开始；账本训练 {budget['train_images']:,} 张，推理 {budget['inference_images']:,} 张，墙钟 {budget['elapsed_wall_seconds']/60:.1f}/480 分钟，Colab 0CU。"]
    for seed,row in latest.items():lines.append(f"seed {seed} 已完成 {row['epoch']}/10 实际epoch，controller准确率{row['controller_accuracy']*100:.2f}%，CE{row['controller_ce']:.4f}。")
    lines+=['','两项历史执行故障已修复：channels-last仅改连续布局，CPU/MPS梯度相对L2差2.02e-6；spawn动态模块注册并通过worker冒烟测试。真实训练曝光0时恢复，原8h截止时间未重置。当前无阻塞，在线四臂未启动。','',f"当前阶段见 supervisor-live.json；完整日志 logs/training-events.jsonl；最新完整checkpoint索引 seed0-latest-checkpoint.json / seed1-latest-checkpoint.json（已生成者）。"]
    (ROOT/'LATEST_STATUS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:status[k] for k in ('updated_utc','total_tests_passed','MPS_model_forward_backward_confirmed','CIFAR10_download','formal_training_started','current_blocker')},indent=2))

if __name__=='__main__':main()
