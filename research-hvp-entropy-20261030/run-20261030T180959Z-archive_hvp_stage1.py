"""Non-destructive archive and delivery of the exact blocked run; no model execution."""
from portable_dev.paths import project_path as _portable_project_path
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone

WORK = pathlib.Path("/Users/lkc/Documents/Codex/2026-10-31/task-3")
SRC = WORK / "research-hvp-entropy-20261030"
RUN = "run-20261030T180959Z"
ARCHIVE = WORK / ("hvp-stage1-" + RUN + ".tar")
PROJECT = _portable_project_path(__file__, 'research-hvp-entropy-20261030')
DEST = PROJECT / RUN
LIMIT = 3*1024**3

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def inventory(root):
    result = {}
    for p in sorted(root.rglob("*")):
        assert not p.is_symlink()
        if p.is_file():
            result[p.relative_to(root).as_posix()] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    return result

if __name__ == "__main__":
    started, cpu_started = time.monotonic(), time.process_time()
    if sys.argv[1:] == ["build"]:
        records = inventory(SRC)
        assert not ARCHIVE.exists()
        assert sum(x["bytes"] for x in records.values())*4 < LIMIT
        with tarfile.open(ARCHIVE, "x") as tf:
            for relative in records:
                tf.add(SRC / relative, arcname=RUN + "/" + relative, recursive=False)
        verified = 0
        with tarfile.open(ARCHIVE, "r") as tf:
            members = tf.getmembers()
            assert len(members) == len(records)
            for member in members:
                assert member.isfile()
                relative = member.name.removeprefix(RUN + "/")
                assert relative in records and member.size == records[relative]["bytes"]
                with tf.extractfile(member) as f:
                    assert hashlib.file_digest(f, "sha256").hexdigest() == records[relative]["sha256"]
                verified += 1
        receipt = {"utc": datetime.now(timezone.utc).isoformat(), "archive": str(ARCHIVE),
                   "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": digest(ARCHIVE),
                   "verified_regular_members": verified, "all_member_hashes_match": True,
                   "run_inventory": records, "packaging_wall_seconds": time.monotonic()-started,
                   "packaging_cpu_process_seconds": time.process_time()-cpu_started}
        (WORK / "archive_verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps({k:v for k,v in receipt.items() if k != "run_inventory"}, indent=2))
    elif sys.argv[1:] == ["install"]:
        receipt = json.loads((WORK / "archive_verification.json").read_text())
        assert inventory(SRC) == receipt["run_inventory"]
        assert digest(ARCHIVE) == receipt["archive_sha256"]
        assert not DEST.exists()
        archive_dest = PROJECT / ARCHIVE.name
        assert not archive_dest.exists()
        size_projected = 2*(sum(x["bytes"] for x in receipt["run_inventory"].values()) + ARCHIVE.stat().st_size) + 1024**2
        assert size_projected < LIMIT
        PROJECT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SRC, DEST)
        shutil.copy2(ARCHIVE, archive_dest)
        assert inventory(DEST) == receipt["run_inventory"]
        assert digest(archive_dest) == receipt["archive_sha256"]
        final = {"utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(DEST),
                 "archive_path": str(archive_dest), "archive_sha256": receipt["archive_sha256"],
                 "all_copied_files_sha256_verified": True, "verified_file_count": len(receipt["run_inventory"]),
                 "archive_members_verified": receipt["verified_regular_members"],
                 "new_disk_bytes_including_staging_copy_and_archives_upper_bound": size_projected,
                 "disk_limit_bytes": LIMIT, "install_wall_seconds": time.monotonic()-started,
                 "install_cpu_process_seconds": time.process_time()-cpu_started,
                 "first_download_to_verified_delivery_seconds": time.time()-json.loads((SRC/"first_download.json").read_text())["epoch"],
                 "scientific_status": "ASSET/VALIDATION-BLOCKED", "mps_seconds": 0,
                 "no_commit_push_upload": True}
        final["packaging_cpu_process_seconds"] = receipt["packaging_cpu_process_seconds"]
        with (PROJECT / (RUN + "-DELIVERY.json")).open("x") as f:
            json.dump(final, f, indent=2)
            f.write("\n")
        shutil.copy2(WORK / "archive_verification.json", PROJECT / (RUN + "-ARCHIVE-VERIFICATION.json"))
        shutil.copy2(pathlib.Path(__file__), PROJECT / (RUN + "-archive_hvp_stage1.py"))
        index = PROJECT / "README.md"
        if not index.exists():
            index.write_text(f"# HVP-DINO stage 1\n\n状态：ASSET/VALIDATION-BLOCKED。指定官方 checkpoint 缺少必需 dino_loss.center；未下载数据、未启用 MPS。\n\n[中文报告]({RUN}/REPORT_ZH.md) · [资产门证据]({RUN}/asset_gate.json) · [交付核验]({RUN}-DELIVERY.json)\n\n目录保留原始权重、官方来源、冻结协议、独立合成数值测试和哈希核验归档；未 commit/push/公开上传。此资产阻塞不等于方向科学否定。\n")
        print(json.dumps(final, indent=2))
    else:
        raise SystemExit("Expected build or install")
