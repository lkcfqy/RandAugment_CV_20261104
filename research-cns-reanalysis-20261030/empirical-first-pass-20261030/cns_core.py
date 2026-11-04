"""CPU-only trajectory logic. No model imports, network, or training entry points."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from typing import Iterable

SCALES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
SHIFTS = ("cartoon_style", "heavy_snow")
MODELS = ("resnet18.a1_in1k", "vit_base_patch16_224.augreg_in1k")


class SchemaError(ValueError):
    pass


def integer(value: str, field: str) -> int:
    try:
        d = Decimal(str(value))
        if not d.is_finite() or d != d.to_integral_value():
            raise ValueError
        return int(d)
    except (InvalidOperation, ValueError):
        raise SchemaError(f"Invalid integer in {field}") from None


def correctness(value: str) -> bool:
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise SchemaError("Correctness must be an explicit boolean")


@dataclass(frozen=True)
class Row:
    shift: str
    class_id: int
    seed: int
    scale: float
    pred_class: int
    correct: bool
    class_str: str
    class_name: str
    img_name: str

    @property
    def key(self):
        return self.shift, self.class_id, self.seed, self.scale

    @property
    def source(self):
        return self.class_id, self.seed


def read_csv_rows(handle, model: str) -> list[Row]:
    reader = csv.DictReader(handle)
    required = {"pred_class", "gt_class", "shift", "class_str", "img_name",
                "class_name", "seed", "scale", model}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise SchemaError(f"Required CSV fields absent for {model}")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise SchemaError("Duplicate CSV column names")
    rows = []
    for record in reader:
        if None in record or any(record[k] is None or record[k] == "" for k in required):
            raise SchemaError("Missing or surplus CSV field values")
        shift = record["shift"]
        if shift not in SHIFTS:
            raise SchemaError(f"Unexpected shift: {shift}")
        try:
            scale = float(Decimal(record["scale"]))
        except (InvalidOperation, ValueError):
            raise SchemaError("Invalid scale") from None
        if scale not in SCALES:
            raise SchemaError("Scale is outside the fixed six observations")
        gt, pred = integer(record["gt_class"], "gt_class"), integer(record["pred_class"], "pred_class")
        if not (0 <= gt < 1000 and 0 <= pred < 1000):
            raise SchemaError("ImageNet class index is outside [0,999]")
        flag = correctness(record[model])
        if flag != (gt == pred):
            raise SchemaError("Correctness disagrees with pred_class/gt_class")
        rows.append(Row(shift, gt, integer(record["seed"], "seed"), scale, pred, flag,
                        record["class_str"], record["class_name"], record["img_name"]))
    return rows


def trajectory_features(sequence: Iterable[bool]) -> dict:
    y = tuple(sequence)
    if len(y) != 6 or any(type(v) is not bool for v in y):
        raise SchemaError("Exactly six explicit booleans are required")
    failed = [i for i, value in enumerate(y) if not value]
    first = failed[0] if failed else None
    terminal = None if y[-1] else (max((i for i, v in enumerate(y) if v), default=-1) + 1)
    opportunities = 0 if first is None else 5 - first
    return {
        "first_error": None if first is None else SCALES[first],
        "terminal_error": None if terminal is None else SCALES[terminal],
        "first_observed_correct": y[0],
        "endpoint_correct": y[-1],
        "all_correct": not failed,
        "correct_to_wrong": sum(a and not b for a, b in zip(y, y[1:])),
        "wrong_to_correct": sum(not a and b for a, b in zip(y, y[1:])),
        "recovered_after_first_error": first is not None and any(y[first + 1:]),
        "recovery_opportunities": opportunities,
        "recovery_eligible": first is not None and opportunities > 0,
        "official_margin_convention": 3.0 - (3.0 if first is None else SCALES[first]),
    }


@dataclass(frozen=True)
class Trajectory:
    model: str
    shift: str
    source: tuple[int, int]
    rows: tuple[Row, ...]

    @property
    def sequence(self):
        return tuple(row.correct for row in self.rows)

    @property
    def features(self):
        return trajectory_features(self.sequence)


def assemble(rows: list[Row], model: str):
    by_key = {}
    classes = {}
    groups = defaultdict(list)
    for row in rows:
        if row.key in by_key:
            raise SchemaError(f"Duplicate key (including possible variants): {row.key}")
        if row.scale not in SCALES or row.shift not in SHIFTS:
            raise SchemaError("Unexpected scale or shift")
        names = (row.class_str, row.class_name)
        if row.class_id in classes and classes[row.class_id] != names:
            raise SchemaError("Class metadata is inconsistent")
        classes[row.class_id] = names
        by_key[row.key] = row
        groups[(row.shift, row.source)].append(row)
    complete, excluded = {}, []
    for (shift, source), group in groups.items():
        ordered = tuple(sorted(group, key=lambda r: r.scale))
        if tuple(r.scale for r in ordered) != SCALES:
            excluded.append({"model": model, "shift": shift, "class_id": source[0],
                             "seed": source[1], "observed_scales": [r.scale for r in ordered],
                             "reason": "incomplete_trajectory_not_censoring"})
            continue
        complete[(shift, source)] = Trajectory(model, shift, source, ordered)
    return complete, excluded


def align_models(rows_by_model: dict[str, list[Row]]) -> dict:
    """Outer key alignment; never align by position. Preserve missingness in audit."""
    tables = {}
    for model, rows in rows_by_model.items():
        assemble(rows, model)  # Enforce uniqueness before converting to dictionaries.
        tables[model] = {row.key: row for row in rows}
    all_keys = set().union(*(set(table) for table in tables.values()))
    for key in all_keys:
        present = [table[key] for table in tables.values() if key in table]
        metadata = {(r.class_str, r.class_name, r.img_name) for r in present}
        if len(metadata) != 1:
            raise SchemaError("Cross-model metadata disagree at the same key")
    return {"outer_union_rows": len(all_keys),
            "rows_per_model": {m: len(t) for m, t in tables.items()},
            "missing_key_counts": {m: len(all_keys - set(t)) for m, t in tables.items()}}


def paired_sources(trajectories: dict) -> tuple[set, set, dict]:
    by_shift = {s: {key for shift, key in trajectories if shift == s} for s in SHIFTS}
    common = by_shift[SHIFTS[0]] & by_shift[SHIFTS[1]]
    eligible = {key for key in common if all(trajectories[(s, key)].sequence[0] for s in SHIFTS)}
    mismatch = [key for key in common if trajectories[(SHIFTS[0], key)].rows[0].pred_class
                != trajectories[(SHIFTS[1], key)].rows[0].pred_class]
    return common, eligible, {"complete_per_shift": {s: len(v) for s, v in by_shift.items()},
                              "common_complete": len(common), "common_first_observed_correct": len(eligible),
                              "scale0_prediction_mismatch_count": len(mismatch),
                              "scale0_prediction_mismatch_keys": sorted(mismatch),
                              "same_predictions_prove_identity": False}


def official_histogram(trajectories: Iterable[Trajectory]) -> dict:
    trajectories = list(trajectories)
    n = len(trajectories)
    counts = {scale: 0 for scale in SCALES}
    for trajectory in trajectories:
        event = trajectory.features["first_error"]
        if event is not None:
            counts[event] += 1
    return {"denominator_all_trajectories": n, "counts": counts,
            "proportions": {s: c / n if n else None for s, c in counts.items()}}
