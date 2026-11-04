"""Package this blocked run truthfully; never load weights or start an experiment."""
import csv
import hashlib
import json
import pathlib
import time
from datetime import datetime, timezone

root = pathlib.Path(__file__).resolve().parent
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def save(name, value):
    (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")

gate = json.loads((root / "asset_gate.json").read_text())
tests = json.loads((root / "test_score.json").read_text())
first = json.loads((root / "first_download.json").read_text())
receipts = json.loads((root / "asset_download_receipts.json").read_text())
assert gate["status"] == "ASSET/VALIDATION-BLOCKED"
assert gate["load_attempts"] == 1 and gate["safe_deserialization"] == "passed"
assert gate["top_level_keys"] == ["early_stopping_epoch", "student", "teacher"]
assert gate["exception_type"] == "KeyError" and gate["error"] == "'dino_loss'"
assert tests["passed"]
for item in receipts:
    path = root / item["relative_path"]
    assert path.stat().st_size == item["bytes"]
    assert digest(path) == item["sha256_local"]

now = datetime.now(timezone.utc).isoformat()
status = {"status": "ASSET/VALIDATION-BLOCKED", "reason_code": "ASSET_BLOCKED",
          "specific_blocker": "Specified official checkpoint lacks dino_loss and therefore the required dino_loss.center [1,65536].",
          "action": "One torch.load(checkpoint,map_location='cpu',weights_only=True), then required tensor/center validation",
          "target": receipts[-1]["requested_url"], "action_outcome": "executed; safe load succeeded; required-center check failed with known outcome",
          "permission_denied": False, "pending_approval": False, "uncertain_outcome": False,
          "real_images": 0, "real_pools": 0, "mps_enabled": False, "backward_calls": 0, "training_steps": 0,
          "data_downloaded": False, "strict_model_load_completed": False,
          "model_forward_completed": False, "throughput_gate_completed": False,
          "real_negative_controls_completed": False, "bootstrap_completed": False,
          "scientific_conclusion": "Undetermined. No evidence for or against a material entropy effect; no contribution established.",
          "stage2_or_stage3_started": False, "reported_utc": now}
save("STATUS.json", status)
with (root / "results.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["status", "temperature", "n_images", "n_pools", "mean_delta_c", "RH", "RH_ci_low", "RH_ci_high", "strict_change_rate", "strict_change_ci_low", "strict_change_ci_high", "real_negative_control", "reason"])
    for temp in (0.04, 0.07):
        writer.writerow([status["status"], temp, 0, 0, "", "", "", "", "", "", "", "NOT_RUN", "missing dino_loss.center"])
with (root / "synthetic_tests.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(tests["checks"][0]))
    writer.writeheader()
    writer.writerows(tests["checks"])
elapsed = time.time() - first["epoch"]
ledger = {"first_download_utc": first["utc"], "report_utc": now,
          "first_download_to_report_seconds": elapsed, "wall_limit_seconds": 7200,
          "cpu_validation_recorded_process_seconds": gate["cpu_process_seconds"] + tests["cpu_process_seconds"],
          "cpu_validation_recorded_wall_seconds": gate["validation_wall_seconds"] + tests["wall_seconds"],
          "cpu_validation_note": "Explicit gate+synthetic scoring timers. Initial interpreter/version check took <1 sec; hash/packaging clocks recorded separately by archive verification. No model CPU forward.",
          "cpu_validation_limit_seconds": 900, "mps_enabled": False, "mps_total_seconds": 0,
          "mps_limit_seconds": 2100, "downloaded_bytes": sum(r["bytes"] for r in receipts),
          "checkpoint_bytes": receipts[-1]["bytes"], "dataset_bytes": 0, "disk_limit_bytes": 3*1024**3,
          "mps_throughput": "NOT_RUN: required asset gate failed", "gpu_purchase": False,
          "scope": "DINO sources and checkpoint only; no Caltech images, no HVP code, no GPU execution"}
save("budget_ledger.json", ledger)
report = f'''# HVP-DINO stage 1：资产门阻塞报告

最终状态：**ASSET/VALIDATION-BLOCKED**（原因码 `ASSET_BLOCKED`）。生成时间：{now}。

指定官方权重安全加载成功，但缺少协议必需的 `dino_loss.center`。本次没有得到真实图像上的熵效应结果，不能判为支持或否定该方向，也没有建立研究贡献。这一关闭仅针对该冻结 pilot；其他探索线由主任务继续协调。

## 已核事实与精确阻塞

- 官方来源为 [facebookresearch/dino 固定提交 README](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/README.md)。README 指向本次 [ViT-S/16 完整 checkpoint](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth)。固定提交 `7c446df5b9f45747937fb0d72314eb9f7b66930a`。
- HTTPS HEAD 与 GET 最终域均为 `dl.fbaipublicfiles.com`，HTTP 200，无跨域跳转。长度与实际文件均为 **352,247,243 bytes**。
- 本地 SHA256：`{gate['sha256_local']}`。这是本地下载后计算并复核的哈希，**不是作者公布哈希**；未把 S3 ETag 当 SHA256。
- 唯一加载调用：`torch.load(path, map_location="cpu", weights_only=True)`。共执行 **1 次**，安全反序列化成功。不是 Namespace allowlist 阻塞。
- 实际顶层键恰为 `early_stopping_epoch`、`student`、`teacher`；访问 `checkpoint["dino_loss"]["center"]` 抛出 `KeyError: 'dino_loss'`。该必需状态不存在，不能补零、估计、拼接其他模型中心或更换加载策略。
- student/teacher 张量有限性与规定 head 形状的代码已先运行到中心检查；完整模型的 `strict=True` 加载尚未执行，不能把本次称为完整模型验证通过。详见 `asset_gate.py` 和 `asset_gate.json` 的实测调用与堆栈。
- [官方 args](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt)已下载核对：原生 teacher temperature=.07、800 epochs、out_dim=65536、use_bn_in_head=false、norm_last_layer=false。README、Apache 2.0 LICENSE、原生 args 原样保留并哈希。

这次操作的结果已知：**加载已执行且成功，必需中心检查已执行且失败**。没有待审批动作，没有不确定的在途 GPU 作业；不是权限审核拒绝造成实验阻塞。未执行不安全加载、globals 扩展或 TorchHub。

## 数据、模型与实验执行边界

按照“先权重门，再数据”的顺序，Caltech101 zip 和图片均**未下载**，因此不存在数据文件哈希或1010图样本清单；`data_provenance.json` 显式记录缺失，未伪造产物。已查 [Caltech 官方记录页](https://data.caltech.edu/records/mzrjq-6wc02)：v1.0、DOI `10.22002/D1.20086`、137.4 MB、页面公布 MD5 `3138e1922a9193bfa496528edbbc45d0`。页面可读 License 栏为空，当前未独立确认 CC BY 4.0；这不等于禁止研究使用。权重门已失败，本轮不继续数据下载或许可证分支。

MPS 未启用，真实图像数 **0/1010**、真实池 **0/2020**；首16图双draw吞吐门、正式前向、真实负控、bootstrap 均未运行。没有训练、backward 或 stage 2/3。没有复制或执行 HVP 代码，也未下载官方 DINO 模型源码来运行。未验证实际梯度一致性。

## 已完成的独立准备与校验

`PROTOCOL.json` 保存冻结范围、种子编码、采样、评分、负控、bootstrap、门槛和预算。候选表按 PCG64(20261031) 从51,480个组合无放回取128个编号并排序，所有条件共用；表中去重 local 子集数为 **127**。表 SHA256：`{tests['candidate_table_sha256']}`。

`score.py` 为独立数学实现；`test_score.py` 仅用53维合成概率，不使用权重或真实图像，也没有拿合成中心补 checkpoint。两温度下 CE=KL+H 最大绝对误差分别为 **{tests['checks'][0]['identity_max_error']:.3g}**、**{tests['checks'][1]['identity_max_error']:.3g}**；独立 Python 标量实现最大差异不超过 **{max(x['scalar_reference_max_error'] for x in tests['checks']):.3g}**。合成 local 负控严格改变率0，同分优先与1e-6严格容差用例通过。以上只证明该合成评分实现的相应性质，**不能替代真实模型负控、完整样本或贡献证据**。

`results.csv` 中两温度均为0样本，科学指标留空；没有把“未测量”写成效应为0。`synthetic_tests.csv` 与 `test_score.json` 单独标注合成测试。

## 预算与复核

首次下载至本报告：**{elapsed:.2f} 秒**，上限7200秒。显式权重门与合成校验记录 CPU process time 合计 **{ledger['cpu_validation_recorded_process_seconds']:.3f} 秒**、wall time **{ledger['cpu_validation_recorded_wall_seconds']:.3f} 秒**；初始解释器版本检查小于1秒，归档哈希计时另见归档收据。MPS **0秒**，没有消费 GPU 额度。权重约336 MiB，所有交付副本与归档会在3 GiB内核验。

软件：Python 3.14.10、PyTorch 2.13.0、NumPy 2.5.4、Pillow 13.0.0；macOS arm64。沿用项目 `.venv`，未安装或更新依赖。项目及各级上级没有发现 AGENTS.md；读取了 validate-data 技能与相关证据质量规范。使用技能不改变冻结协议。

证据见 `asset_download_receipts.json`、`asset_gate.json`、`STATUS.json`、`budget_ledger.json`、`PROTOCOL.json`、`candidate_table.json/csv`、`test_score.json`、`MANIFEST.sha256` 与归档验证收据。没有 commit、push 或公开上传。保留原有实验历史。

## 后续边界

本冻结 pilot 在资产门关闭；缺中心不能用“继续找贡献”的总授权绕过本轮硬约束。只有主任务另行审查可用完整中心的官方资产或重新冻结其他设计，才可开启新实验。本报告不擅自选择替代 checkpoint，不启动后续阶段。
'''
(root / "REPORT_ZH.md").write_text(report)
save("data_provenance.json", {"official_record_url": "https://data.caltech.edu/records/mzrjq-6wc02", "doi": "10.22002/D1.20086", "version": "1.0", "published_zip_size_display": "137.4 MB", "published_md5": "3138e1922a9193bfa496528edbbc45d0", "downloaded": False, "local_md5": None, "local_sha256": None, "sample_manifest": None, "license": "Not independently confirmed; browser-readable License section empty. Not a determination of prohibition.", "gate": "Weight gate failed first; data branch not entered"})
(root / "README.md").write_text("# HVP-DINO stage 1 frozen audit\n\nStatus: ASSET/VALIDATION-BLOCKED. Required official checkpoint center is missing. No real-image or MPS experiment ran.\n\nRead REPORT_ZH.md, STATUS.json and asset_gate.json first. MANIFEST.sha256 covers the preserved evidence. The checkpoint SHA256 is locally calculated, not author-published. No data was downloaded. Do not rerun asset_gate.py in this run; its one-attempt safety guard is intentional. score.py/test_score.py are small independent mathematical controls, not a complete or authorized alternative experimental pipeline.\n")
paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"MANIFEST.sha256", "manifest_verification.json"})
manifest = "".join(f"{digest(p)}  {p.relative_to(root).as_posix()}\n" for p in paths)
(root / "MANIFEST.sha256").write_text(manifest)
save("manifest_verification.json", {"utc": datetime.now(timezone.utc).isoformat(), "file_count": len(paths), "all_file_hashes_recomputed": True, "manifest_sha256": digest(root / "MANIFEST.sha256"), "run_file_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()), "protocol_sha256": digest(root / "PROTOCOL.json"), "status": status["status"]})
print(json.dumps({"status": status["status"], "wall_seconds": elapsed, "files_hashed": len(paths), "report": str(root / "REPORT_ZH.md")}, indent=2))
