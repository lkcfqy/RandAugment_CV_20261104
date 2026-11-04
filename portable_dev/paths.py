"""Locate this checkout without retaining the original machine's directory layout.

These helpers locate files; they do not download resources or relax historical
hash, budget, or sealing checks. Asset overrides are opt-in local directories.
"""
from __future__ import annotations

import os
from pathlib import Path


def _checked_root(value: str | Path, *, project: bool) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Configured resource root must be an existing directory")
    if project and not ((root / "portable_dev" / "paths.py").is_file() and (root / "README.md").is_file()):
        raise ValueError("RANDAUGMENT_PROJECT_ROOT must point to this portable checkout")
    return root


def project_root(origin: str | Path) -> Path:
    """Find the checkout from a source file, or use an explicit validated override."""
    override = os.environ.get("RANDAUGMENT_PROJECT_ROOT")
    if override is not None:
        return _checked_root(override, project=True)
    source = Path(origin).resolve()
    start = source if source.is_dir() else source.parent
    for root in (start, *start.parents):
        if (root / "portable_dev" / "paths.py").is_file() and (root / "README.md").is_file():
            return root
    raise ValueError("Could not locate the portable checkout from this source file")


def _within(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Resource paths must be relative to their configured root")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Resource path escapes its configured root")
    return result


def project_path(origin: str | Path, relative: str | Path = "") -> Path:
    return _within(project_root(origin), relative)


def asset_path(origin: str | Path, relative: str | Path) -> Path:
    override = os.environ.get("RANDAUGMENT_ASSET_ROOT")
    root = _checked_root(override, project=False) if override is not None else project_root(origin)
    return _within(root, relative)
