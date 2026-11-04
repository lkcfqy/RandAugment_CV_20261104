"""Independent mathematical implementation; no third-party project code copied."""
import math
import numpy as np


def softmax(logits, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64) / temperature
    x = x - x.max(axis=-1, keepdims=True)
    v = np.exp(x)
    return v / v.sum(axis=-1, keepdims=True)


def rank_spectrum(prob):
    prob = np.asarray(prob, dtype=np.float64)
    assert np.isfinite(prob).all() and (prob >= 0).all()
    assert np.allclose(prob.sum(axis=-1), 1, rtol=0, atol=5e-15)
    order = np.argsort(-prob, axis=-1, kind='stable')
    spectrum = np.take_along_axis(prob, order, axis=-1)
    return order, spectrum


def arms(p, q):
    op, vp = rank_spectrum(p)
    oq, vq = rank_spectrum(q)
    return {'00': (op, vp), '10': (oq, vp), '01': (op, vq), '11': (oq, vq)}


def scores_sorted(spectrum, u, method='aps', lam=0.0, kreg=0):
    v = np.asarray(spectrum, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)[:, None]
    assert v.ndim == 2 and u.shape[0] == v.shape[0]
    assert ((u >= 0) & (u < 1)).all()
    r = np.arange(1, v.shape[1] + 1, dtype=np.float64)[None, :]
    if method in ('aps', 'raps'):
        # Exclusive sum avoids subtracting tiny tail probabilities from 1.
        before = np.concatenate([np.zeros((len(v), 1)), np.cumsum(v[:, :-1], axis=1)], axis=1)
        out = before + u * v
        if method == 'raps':
            out = out + lam * np.maximum(r - kreg, 0)
    elif method == 'saps':
        # Huang et al. ICML 2024 Eq. (10): rank 1 U*pmax;
        # rank > 1: pmax + lambda*(rank-2+U).
        out = v[:, :1] + lam * (r - 2 + u)
        out[:, 0] = (u * v[:, :1])[:, 0]
    elif method == 'rankonly':
        out = np.broadcast_to(r - 1 + u, v.shape).copy()
    else:
        raise ValueError(method)
    assert np.isfinite(out).all()
    assert (np.diff(out, axis=1) >= -1e-14).all()
    return out


def label_scores(order, sorted_scores):
    out = np.empty_like(sorted_scores)
    np.put_along_axis(out, order, sorted_scores, axis=1)
    return out


def quantile(calibration_scores, alpha):
    a = np.asarray(calibration_scores, dtype=np.float64)
    assert a.ndim == 1 and len(a) > 0 and np.isfinite(a).all()
    k = math.ceil((len(a) + 1) * (1 - alpha))
    return float('inf') if k > len(a) else float(np.partition(a, k - 1)[k - 1])


def calibrate_predict(order, spectrum, u, y, ncal, alpha, **kwargs):
    scores = label_scores(order, scores_sorted(spectrum, u, **kwargs))
    threshold = quantile(scores[np.arange(ncal), y[:ncal]], alpha)
    return scores[ncal:] <= threshold, threshold


def coverage_metrics(covered, y, classes=100, target=.9):
    n = np.bincount(y, minlength=classes)
    successes = np.bincount(y, weights=covered, minlength=classes)
    if (n < 2).any():
        return {'coverage': float(np.mean(covered)), 'D': None, 'Dhat': None, 'class_n': n.tolist(), 'reason': 'At least one class has n<2'}
    c = successes / n
    return {'coverage': float(np.mean(covered)), 'D': float(np.mean((c-target)**2)),
            'Dhat': float(np.mean((c-target)**2 - c*(1-c)/(n-1))),
            'class_n': n.tolist(), 'class_coverage': c.tolist()}
