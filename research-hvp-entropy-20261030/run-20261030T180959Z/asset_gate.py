"""Single CPU weights-only load. No allowlisting, unsafe fallback, or GPU API."""
import hashlib
import json
import pathlib
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
if (ROOT / "asset_gate.json").exists():
    raise SystemExit("Refusing a second safe-load attempt in the same run")
BEGIN = time.monotonic()
CPU_BEGIN = time.process_time()
import torch
import numpy as np
import PIL

torch.set_num_threads(2)
protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
receipt = {"utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__, "numpy": np.__version__,
           "Pillow": PIL.__version__, "python": sys.version, "platform": platform.platform(),
           "load_call": "torch.load(path, map_location='cpu', weights_only=True)", "load_attempts": 0,
           "mps_enabled": False, "dataset_downloaded": False, "state_dict_strict_load": "not yet attempted"}
path = ROOT / "assets/dino_deitsmall16_pretrain_full_checkpoint.pth"
try:
    recorded = json.loads((ROOT / "asset_download_receipts.json").read_text())[-1]
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    assert digest == recorded["sha256_local"] and path.stat().st_size == recorded["bytes"]
    receipt["sha256_local"] = digest
    receipt["load_attempts"] = 1
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict), "Top-level checkpoint must be a dict"
    receipt["top_level_keys"] = sorted(checkpoint.keys())
    checks = {}
    requirements = {k: v for k, v in protocol["checkpoint_requirements"].items() if k.startswith("head.")}
    for model in ("student", "teacher"):
        state = checkpoint[model]
        assert isinstance(state, dict)
        normalized = {}
        removed = 0
        for key, tensor in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
                removed += 1
            assert key not in normalized, "Prefix removal key collision"
            assert isinstance(tensor, torch.Tensor), "Non-tensor in state dict"
            assert torch.isfinite(tensor).all().item(), "Non-finite state: " + key
            normalized[key] = tensor
        for key, shape in requirements.items():
            assert list(normalized[key].shape) == shape, f"Wrong {model} {key} shape"
        checks[model] = {"keys": len(normalized), "prefixes_removed": removed,
                         "head_shapes": {k: list(normalized[k].shape) for k in requirements}, "all_finite": True}
    center = checkpoint["dino_loss"]["center"]
    assert isinstance(center, torch.Tensor) and list(center.shape) == [1, 65536]
    assert torch.isfinite(center).all().item()
    receipt.update(safe_deserialization="passed", tensor_checks=checks, center_shape=list(center.shape),
                   status="SAFE_TENSORS_PASSED_PENDING_MODEL_STRICT_LOAD")
except Exception as exc:
    receipt.update(status="ASSET/VALIDATION-BLOCKED", reason_code="ASSET_BLOCKED", exception_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc(), safe_deserialization="failed" if "checkpoint" not in globals() else "passed")
finally:
    receipt["cpu_process_seconds"] = time.process_time() - CPU_BEGIN
    receipt["validation_wall_seconds"] = time.monotonic() - BEGIN
    (ROOT / "asset_gate.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
