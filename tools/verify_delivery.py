#!/usr/bin/env python3
"""Read-only check of this development delivery, not the original experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path) -> dict:
    manifest_path = root / "PORTABLE_DELIVERY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported delivery manifest schema")
    failures = []
    verified = 0
    total_bytes = 0
    for item in manifest["files"]:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            failures.append({"path": str(rel), "error": "invalid relative path"})
            continue
        path = root.joinpath(*rel.parts)
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
            failures.append({"path": str(rel), "error": "symlink is not a delivered regular file"})
            continue
        if not path.is_file():
            failures.append({"path": str(rel), "error": "missing file"})
            continue
        if path.stat().st_size != item["bytes"]:
            failures.append({"path": str(rel), "error": "size mismatch"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            failures.append({"path": str(rel), "error": "sha256 mismatch"})
            continue
        verified += 1
        total_bytes += item["bytes"]
    return {
        "scope": "Files in the current development delivery; excludes the manifest itself.",
        "original_experiments_fully_restored": False,
        "verified_files": verified,
        "verified_bytes": total_bytes,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
