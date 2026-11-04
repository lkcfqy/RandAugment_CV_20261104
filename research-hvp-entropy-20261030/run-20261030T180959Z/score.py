"""Independent frozen CE/KL/H scoring; no model loading or autograd."""
import itertools
import json
import hashlib
import numpy as np

GLOBAL_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))

def candidate_table():
    local = list(itertools.combinations(range(16), 8))
    ids = np.sort(np.random.Generator(np.random.PCG64(20261031)).choice(4 * len(local), 128, replace=False))
    table = [{"id": int(i), "a": GLOBAL_PAIRS[i // len(local)][0], "b": GLOBAL_PAIRS[i // len(local)][1],
              "locals": list(local[i % len(local)])} for i in ids]
    return table

def view_seed(relative_path, draw, view_id):
    encoded = json.dumps([20261030, relative_path, draw, view_id], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")

def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

def pair_scores(teacher_logits, student_logits, center, temperature):
    logq = log_softmax((np.asarray(teacher_logits, dtype=np.float64) - center) / temperature)
    logp = log_softmax(np.asarray(student_logits, dtype=np.float64) / 0.1)
    q = np.exp(logq)
    entropy = -np.sum(q * logq, axis=-1)
    ce = -(q @ logp.T)
    # Independent direct KL computation, deliberately not ce - entropy.
    kl = np.empty_like(ce)
    for i in range(q.shape[0]):
        kl[i] = np.sum(q[i] * (logq[i] - logp), axis=-1)
    assert np.isfinite(ce).all() and np.isfinite(kl).all() and np.isfinite(entropy).all()
    return ce, kl, entropy

def candidate_scores(ce, kl, entropy, table):
    values = []
    for item in table:
        a, b = item["a"], item["b"]
        local = np.asarray(item["locals"]) + 4
        c = (ce[a,b] + ce[b,a] + ce[a,local].sum() + ce[b,local].sum()) / 18
        k = (kl[a,b] + kl[b,a] + kl[a,local].sum() + kl[b,local].sum()) / 18
        values.append((c, k, (entropy[a] + entropy[b]) / 2))
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    return values

def summarize(values):
    c, k, h = values.T
    wc, wk = int(c.argmax()), int(k.argmax())
    c_gap, k_gap = float(c[wc] - c[wk]), float(k[wk] - k[wc])
    delta = values[wc] - values.mean(axis=0)
    return {"winner_ce_position": wc, "winner_kl_position": wk,
            "raw_changed": wc != wk, "strict_changed": bool(wc != wk and c_gap > 1e-6 and k_gap > 1e-6),
            "ce_gap": c_gap, "kl_gap": k_gap, "delta_c": float(delta[0]), "delta_k": float(delta[1]),
            "delta_h": float(delta[2]), "identity_max_error": float(np.abs(c - k - h).max())}

def negative_table(table):
    return [{"a": 0, "b": 2, "locals": list(s)} for s in sorted(set(tuple(item["locals"]) for item in table))]
