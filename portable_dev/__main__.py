"""Run only the explicit synthetic-development suite; never discover experiment runners."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .paths import project_root, asset_path

# Each test module was statically reviewed for import and execution side effects.
# In particular, do NOT discover test_second_encoder_contract.py or run sealed run.py.
SUITES = (
    ("path configuration", "", ("-m", "unittest", "-v", "portable_dev.tests.test_paths")),
    ("CIFAR synthetic persistence and seal guards", "augmentation-selection-cifar100-run-20261102", ("-m", "unittest", "-v", "test_integration")),
    ("selection mathematics", "augmentation-selection-synthesis-20261102", ("-m", "unittest", "-v", "test_analysis")),
    ("CNS synthetic identities and statistics", "research-cns-reanalysis-20261030", ("-m", "unittest", "-v", "test_cns_core", "test_cns_statistics")),
    ("strict integer accounting", "augmentation-selection-cifar100-run-20261102", ("exact_accounting.py",)),
    ("complete tie bounds", "augmentation-selection-cifar100-preflight-20261102", ("compact_tie_bounds.py",)),
)
EXTERNAL_ASSETS = (
    ("excluded_by_user", "augmentation-selection-cifar100-run-20261102/dinov2_small/W.npy"),
    ("excluded_by_user", "augmentation-selection-cifar100-run-20261102/dinov2_small/support_features.npy"),
    ("excluded_by_user", "research-frofa-pilot-20261030/cache/features.fp32.npy"),
    ("excluded_by_user", "research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy"),
    ("model_weight", "research-shared-theta-ad-20261030/models/dinov2-small/model.safetensors"),
    ("model_weight", "research-view-route-20261030/assets/deit_tiny_patch16_224-a1311bcf.pth"),
    ("dataset_archive", "research-frofa-pilot-20261030/assets/EuroSAT_RGB.zip"),
)


def doctor(root: Path) -> dict:
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    sources = [
        "portable_dev/tests/test_paths.py",
        "augmentation-selection-cifar100-run-20261102/test_integration.py",
        "augmentation-selection-cifar100-run-20261102/contracts.py",
        "augmentation-selection-cifar100-run-20261102/exact_accounting.py",
        "augmentation-selection-synthesis-20261102/test_analysis.py",
        "augmentation-selection-synthesis-20261102/analyze_saved.py",
        "research-cns-reanalysis-20261030/test_cns_core.py",
        "research-cns-reanalysis-20261030/test_cns_statistics.py",
        "research-cns-reanalysis-20261030/cns_core.py",
        "research-cns-reanalysis-20261030/cns_statistics.py",
        "augmentation-selection-cifar100-preflight-20261102/compact_tie_bounds.py",
    ]
    missing_source = [p for p in sources if not (root / p).is_file()]
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": numpy_version,
        "development_dependencies_ready": sys.version_info >= (3, 12) and numpy_version == "2.5.4" and not missing_source,
        "missing_development_sources": missing_source,
        "assets": [{"kind": kind, "relative_path": path, "present": asset_path(__file__, path).is_file()} for kind, path in EXTERNAL_ASSETS],
        "historical_experiment_reproduction": "NOT_VERIFIED: excluded assets, original hash seals and historical deadlines remain in force",
        "scope": "Only explicit CPU synthetic tests. No model, network, dataset download, cloud or sealed experiment entry point.",
    }


def test(root: Path) -> int:
    check = doctor(root)
    print(json.dumps(check, indent=2, ensure_ascii=False), flush=True)
    if not check["development_dependencies_ready"]:
        print("Install the minimal development requirements in an isolated Python >=3.12 environment.", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "RANDAUGMENT_PROJECT_ROOT": str(root), "PYTHONPATH": str(root)})
    results = []
    for title, relative, args in SUITES:
        print(f"\n=== {title} ===", flush=True)
        command = [sys.executable, "-B", *args]
        completed = subprocess.run(command, cwd=root / relative, env=env, check=False)
        results.append({"suite": title, "exit_code": completed.returncode})
    print(json.dumps({"development_suite_results": results, "historical_reproduction_claimed": False}, indent=2), flush=True)
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "test"))
    args = parser.parse_args()
    try:
        root = project_root(__file__)
        if args.command == "doctor":
            result = doctor(root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["development_dependencies_ready"] else 2
        return test(root)
    except (ValueError, OSError) as error:
        print(f"Development configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
