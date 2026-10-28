"""Offline audit checks; no images, model inference or scientific results.

Only the actual-group CENSUS branch is implemented: every raw-pool candidate
must already have an independently established actual_group. Group labels are
not inferred from prompts. The caller declares that census complete; this
program cannot authenticate its provenance. Missing G needs another estimator.

Target E is label_validity=valid AND background_change_success=1 AND is_copy=0.
These are separate human audit fields. A known failure makes E ineligible;
otherwise incomplete evidence makes E unknown. reviewer_disagreement describes
this target audit, not the separate complete group census.

audit_probability logs target-audit inclusion probability, including for rows
not selected. Output rates on known-E rows are unweighted descriptions, not
population estimates. Bounds are finite-pool worst-case bounds, conditional
on at least one E example, without MAR or IID assumptions.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict


def retention_bounds(a, c, u, v):
    """Known-E kept/rejected a/c, unknown-E kept/rejected u/v."""
    counts = (a, c, u, v)
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("counts must be nonnegative integers")
    if not sum(counts):
        return {"lower": None, "upper": None, "target_exists_guaranteed": False,
                "reason": "No possibly target-eligible example; recall undefined"}
    low_den = a + c + v
    high_den = a + c + u
    return {
        "lower": a / low_den if low_den else 1.0,
        "upper": (a + u) / high_den if high_den else 0.0,
        "target_exists_guaranteed": a + c > 0,
        "condition": "At least one E example exists in the complete group pool",
    }


def binary(value, field, optional=False):
    if optional and value == "":
        return None
    if value not in ("0", "1"):
        raise ValueError(field + " must be 0/1" + ("/blank" if optional else ""))
    return int(value)


def target_eligibility(label_validity, change_success, is_copy):
    if label_validity == "invalid" or change_success == 0 or is_copy == 1:
        return "ineligible"
    if label_validity == "valid" and change_success == 1 and is_copy == 0:
        return "eligible"
    return "unknown"


def gate_table(items):
    table = {}
    for c in (0, 1):
        for n in (0, 1, None):
            subset = [r for r in items if r['c'] == c and r['n'] == n]
            label_counts = Counter(r['label'] for r in subset)
            target_counts = Counter(r['e'] for r in subset)
            table['C'+str(c)+'_N'+('missing' if n is None else str(n))] = {
                "candidate_count": len(subset),
                "retained_count": sum(r['retained'] for r in subset),
                "label_validity": {v: label_counts[v]
                                   for v in ('valid', 'invalid', 'unknown')},
                "target_eligibility": {e: target_counts[e]
                                       for e in ('eligible', 'ineligible', 'unknown')},
                "label_validity_x_target_eligibility": {
                    v: {e: sum(r['label'] == v and r['e'] == e for r in subset)
                        for e in ('eligible', 'ineligible', 'unknown')}
                    for v in ('valid', 'invalid', 'unknown')},
            }
    return table


def analyze(rows, expected_candidates, *, actual_group_census=False):
    if not actual_group_census:
        raise ValueError("Declare independently established full actual-group census first")
    if type(expected_candidates) is not int or expected_candidates < 0:
        raise ValueError("expected_candidates must be a nonnegative integer")
    if len(rows) != expected_candidates:
        raise ValueError("Row count differs from the external raw-pool count")
    ids = set()
    groups = defaultdict(list)
    for row in rows:
        uid = row.get("candidate_id", "").strip()
        source = row.get("source_id", "").strip()
        group = row.get("actual_group", "").strip()
        if not uid or uid in ids:
            raise ValueError("Missing or duplicate candidate_id")
        ids.add(uid)
        if not source:
            raise ValueError("source_id required for source-cluster accounting")
        if not group or group.lower() in ("unknown", "missing", "na"):
            raise ValueError("Actual group unknown: group-specific bounds not identified")
        label = row.get("label_validity")
        if label not in ("valid", "invalid", "unknown"):
            raise ValueError("label_validity must be valid/invalid/unknown")
        success = binary(row.get("background_change_success"),
                         "background_change_success", optional=True)
        copy = binary(row.get("is_copy"), "is_copy", optional=True)
        e = target_eligibility(label, success, copy)
        disagreement = row.get("reviewer_disagreement")
        if disagreement not in ("not_reviewed", "none", "unresolved", "resolved"):
            raise ValueError("reviewer_disagreement must be not_reviewed/none/unresolved/resolved")
        if disagreement == "not_reviewed" and (
                label != "unknown" or success is not None or copy is not None):
            raise ValueError("Unreviewed target fields must remain unknown/blank")
        if disagreement == "unresolved" and e != "unknown":
            raise ValueError("Unresolved target disagreement cannot be counted as known E")
        try:
            pi = float(row.get("audit_probability", ""))
        except (TypeError, ValueError):
            raise ValueError("audit_probability must be a finite number in [0,1]")
        if not math.isfinite(pi) or not 0 <= pi <= 1:
            raise ValueError("audit_probability must be a finite number in [0,1]")
        if disagreement != "not_reviewed" and pi == 0:
            raise ValueError("Reviewed row cannot have zero audit inclusion probability")
        c = binary(row.get("semantic_pass"), "semantic_pass")
        n = binary(row.get("novelty_pass"), "novelty_pass", optional=True)
        retained = binary(row.get("retained"), "retained")
        if c == 0 and retained:
            raise ValueError("Inconsistent gate log: semantic failure was retained")
        if n is not None and retained != c * n:
            raise ValueError("Log inconsistent with two fixed conjunctive gates")
        groups[group].append(dict(source=source, label=label, success=success,
                                  copy=copy, e=e, c=c, n=n, retained=retained,
                                  pi=pi, disagreement=disagreement))
    output = {}
    for group, items in sorted(groups.items()):
        eligible = [r for r in items if r['e'] == "eligible"]
        a = sum(r['retained'] for r in eligible)
        c = len(eligible) - a
        unknown = [r for r in items if r['e'] == "unknown"]
        u = sum(r['retained'] for r in unknown)
        v = len(unknown) - u
        semantic_eligible = [r for r in eligible if r['c']]
        n_complete = all(r['n'] is not None for r in eligible)
        conditional_complete = all(r['n'] is not None for r in semantic_eligible)
        output[group] = {
            "candidate_count": len(items),
            "source_count": len({r['source'] for r in items}),
            "known_target_eligible": len(eligible),
            "known_ineligible": sum(r['e'] == "ineligible" for r in items),
            "target_eligibility_unknown": len(unknown),
            "eligible_kept_a": a, "eligible_rejected_c": c,
            "unknown_kept_u": u, "unknown_rejected_v": v,
            "target_recall_bounds": retention_bounds(a, c, u, v),
            "known_eligible_unweighted_descriptive_rates": {
                "semantic": len(semantic_eligible) / len(eligible) if eligible else None,
                "novelty_all": (sum(r['n'] for r in eligible) / len(eligible)
                                if eligible and n_complete else None),
                "joint": a / len(eligible) if eligible else None,
                "novelty_given_semantic": (
                    sum(r['n'] for r in semantic_eligible) / len(semantic_eligible)
                    if semantic_eligible and conditional_complete else None),
            },
            "stage2_missing_on_known_eligible": sum(r['n'] is None for r in eligible),
            "gate_combinations": gate_table(items),
            "audit_log": {
                "zero_inclusion_probability_count": sum(r['pi'] == 0 for r in items),
                "inclusion_probability_min": min(r['pi'] for r in items),
                "inclusion_probability_max": max(r['pi'] for r in items),
                "reviewer_disagreement": dict(Counter(r['disagreement'] for r in items)),
            },
        }
    return {
        "candidate_count": len(rows), "groups": output,
        "actual_group_census_declared": True,
        "target": "E = label-valid AND background-change-success AND not-copy",
        "inference": "No causal, group-population or downstream-performance conclusion",
        "provenance_limit": "Count/declaration checks cannot prove full-pool or label provenance",
        "uncertainty": "No confidence intervals or design-weighted estimates are computed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--expected-candidates", type=int, required=True,
                        help="Independent complete raw-pool count, not retained count")
    parser.add_argument("--confirm-actual-group-census", action="store_true",
                        help="Declare actual groups independently known for every candidate")
    args = parser.parse_args()
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        result = analyze(rows, args.expected_candidates,
                         actual_group_census=args.confirm_actual_group_census)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
