"""Small CPU-only mathematical controls; these are not scientific observations."""
import csv
import hashlib
import json
import pathlib
import time
import numpy as np
from score import candidate_table, view_seed, pair_scores, candidate_scores, summarize, negative_table

root = pathlib.Path(__file__).resolve().parent
start, cpu = time.monotonic(), time.process_time()
table = candidate_table()
assert len(table) == 128 and len(set(t["id"] for t in table)) == 128
assert [t["id"] for t in table] == sorted(t["id"] for t in table)
assert all(len(t["locals"]) == len(set(t["locals"])) == 8 for t in table)
assert all(0 <= t["id"] < 51480 for t in table)
content = json.dumps(table, separators=(",", ":")) + "\n"
(root / "candidate_table.json").write_text(content)
with (root / "candidate_table.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "a", "b", *[f"local_{i}" for i in range(8)]])
    for t in table:
        writer.writerow([t["id"], t["a"], t["b"], *t["locals"]])
rng = np.random.Generator(np.random.PCG64(9001))
teacher, student = rng.normal(0, 0.1, (4, 53)), rng.normal(0, 0.1, (20, 53))
center = rng.normal(0, 0.01, (1, 53))
checks = []
for temp in (0.04, 0.07):
    ce, kl, h = pair_scores(teacher, student, center, temp)
    scores = candidate_scores(ce, kl, h, table)
    result = summarize(scores)
    neg = summarize(candidate_scores(ce, kl, h, negative_table(table)))
    assert result["identity_max_error"] <= 1e-12
    assert neg["identity_max_error"] <= 1e-12 and not neg["strict_changed"]
    # Scalar reference for every selected CE/KL pair uses independently normalized probabilities.
    from math import exp, log
    qq = [[exp(x) for x in row] for row in (teacher - center) / temp]
    qq = [[x / sum(row) for x in row] for row in qq]
    pp = [[exp(x) for x in row] for row in student / 0.1]
    pp = [[x / sum(row) for x in row] for row in pp]
    max_error = 0.0
    for t, actual in zip(table, scores):
        a, b = t["a"], t["b"]
        pairs = [(a,b), (b,a)] + [(g,l+4) for l in t["locals"] for g in (a,b)]
        c = sum(-sum(q*log(p) for q,p in zip(qq[i],pp[j])) for i,j in pairs)/18
        k = sum(sum(q*log(q/p) for q,p in zip(qq[i],pp[j])) for i,j in pairs)/18
        max_error = max(max_error, abs(c-actual[0]), abs(k-actual[1]))
    assert max_error < 1e-12
    checks.append({"temperature": temp, "identity_max_error": result["identity_max_error"],
                   "scalar_reference_max_error": max_error, "negative_control_strict_changed": neg["strict_changed"]})
assert summarize(np.zeros((128,3)))["winner_ce_position"] == 0
near = np.array([[1.0,0.5,0.5],[1.0+5e-7,0.4,0.6000005]])
assert not summarize(near)["strict_changed"]
seed_example = view_seed("accordion/image_0001.jpg", 0, "A0")
assert seed_example != view_seed("accordion/image_0001.jpg", 1, "A0")
assert seed_example != view_seed("accordion/image_0001.jpg", 0, "A1")
report = {"scope": "synthetic scoring tests only; not real-image/model validation", "passed": True,
          "candidate_table_sha256": hashlib.sha256(content.encode()).hexdigest(),
          "n_negative_subsets": len(negative_table(table)), "checks": checks,
          "tie_test": "passed", "strict_tolerance_test": "passed", "view_seed_example": seed_example,
          "wall_seconds": time.monotonic()-start, "cpu_process_seconds": time.process_time()-cpu}
(root / "test_score.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
