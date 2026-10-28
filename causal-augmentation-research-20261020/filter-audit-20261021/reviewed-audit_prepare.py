"""Offline metadata checks; no images, model inference or scientific results.

Rates computed only on audited-valid rows are descriptive, not population
estimates. Bounds are finite-pool worst-case bounds conditional on at least
one valid example. They need complete candidate enumeration and known actual
groups; this program cannot authenticate upstream provenance or human labels.
"""
import argparse
import csv
import json
from collections import defaultdict


def retention_bounds(a, c, u, v):
    """Known-valid kept/rejected a/c, unknown-validity kept/rejected u/v.

    Invalid audited examples do not enter the valid-recall denominator.
    No missing-at-random or independence assumption is used here.
    """
    counts = (a, c, u, v)
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("counts must be nonnegative integers")
    if not sum(counts):
        return {"lower": None, "upper": None, "valid_exists_guaranteed": False,
                "reason": "No possibly valid example; valid recall undefined"}
    low_den = a + c + v
    high_den = a + c + u
    return {
        "lower": a / low_den if low_den else 1.0,
        "upper": (a + u) / high_den if high_den else 0.0,
        "valid_exists_guaranteed": a + c > 0,
        "condition": "At least one valid example exists in the complete group pool",
    }


def binary(value, field, optional=False):
    if optional and value == "":
        return None
    if value not in ("0", "1"):
        raise ValueError(field + " must be 0/1" + ("/blank" if optional else ""))
    return int(value)


def analyze(rows, expected_candidates):
    if len(rows) != expected_candidates:
        raise ValueError("Row count differs from the external raw-pool count")
    ids = set()
    groups = defaultdict(list)
    for row in rows:
        uid = row.get("candidate_id", "").strip()
        group = row.get("actual_group", "").strip()
        if not uid or uid in ids:
            raise ValueError("Missing or duplicate candidate_id")
        ids.add(uid)
        if not group or group.lower() in ("unknown", "missing", "na"):
            raise ValueError("Actual group unknown: group-specific bounds not identified")
        validity = row.get("validity")
        if validity not in ("valid", "invalid", "unknown"):
            raise ValueError("validity must be valid/invalid/unknown")
        c = binary(row.get("semantic_pass"), "semantic_pass")
        n = binary(row.get("novelty_pass"), "novelty_pass", optional=True)
        retained = binary(row.get("retained"), "retained")
        if c == 0 and retained:
            raise ValueError("Inconsistent gate log: semantic failure was retained")
        if n is not None and retained != c * n:
            raise ValueError("Log inconsistent with two fixed conjunctive gates")
        groups[group].append((validity, c, n, retained))
    output = {}
    for group, items in sorted(groups.items()):
        valid = [r for r in items if r[0] == "valid"]
        a = sum(r[3] for r in valid)
        c = len(valid) - a
        unknown = [r for r in items if r[0] == "unknown"]
        u = sum(r[3] for r in unknown)
        v = len(unknown) - u
        semantic_valid = [r for r in valid if r[1]]
        n_complete = all(r[2] is not None for r in valid)
        conditional_complete = all(r[2] is not None for r in semantic_valid)
        output[group] = {
            "candidate_count": len(items),
            "audited_valid": len(valid),
            "audited_invalid": sum(r[0] == "invalid" for r in items),
            "validity_unknown": len(unknown),
            "valid_kept_a": a, "valid_rejected_c": c,
            "unknown_kept_u": u, "unknown_rejected_v": v,
            "valid_recall_bounds": retention_bounds(a, c, u, v),
            "audited_valid_descriptive_rates": {
                "semantic": len(semantic_valid) / len(valid) if valid else None,
                "novelty_all": (sum(r[2] for r in valid) / len(valid)
                                if valid and n_complete else None),
                "joint": a / len(valid) if valid else None,
                "novelty_given_semantic": (
                    sum(r[2] for r in semantic_valid) / len(semantic_valid)
                    if semantic_valid and conditional_complete else None),
            },
            "stage2_missing_on_audited_valid": sum(r[2] is None for r in valid),
        }
    return {
        "candidate_count": len(rows), "groups": output,
        "inference": "No causal, group-population or downstream-performance conclusion",
        "provenance_limit": "Count check cannot prove complete raw-pool provenance",
        "uncertainty": "No independent-sample confidence intervals are claimed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--expected-candidates", type=int, required=True,
                        help="Independent upstream complete-pool count, not retained count")
    args = parser.parse_args()
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        result = analyze(rows, args.expected_candidates)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
