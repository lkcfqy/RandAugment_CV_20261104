"""Independent CPU mathematical implementation. Never imports audited source."""
from dataclasses import dataclass
import numpy as np


def normalize_channel(x, epsilon=1e-8):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("one finite nonempty channel required")
    lo, hi = x.min(), x.max()
    d = float(hi - lo + epsilon)
    if d <= 0:
        raise ValueError("positive d required; official epsilon is not removed")
    return (x-lo)/d, d, float(lo)


def nodes_weights(u, a, clip=True):
    u = np.asarray(u, dtype=np.float64)
    if a < 0 or not np.isfinite(a):
        raise ValueError("finite nonnegative amplitude required")
    if a == 0:
        return np.array([0.]), np.array([1.]), np.array([0.])
    interior = np.concatenate((-u, 1-u)) if clip else np.array([])
    interior = interior[(interior > -a) & (interior < a)]
    edges = np.unique(np.concatenate(([-a], interior, [a])))
    mid = (edges[:-1]+edges[1:])/2
    half = (edges[1:]-edges[:-1])/2
    q = np.stack((mid-half/np.sqrt(3.), mid+half/np.sqrt(3.)), axis=1).ravel()
    w = np.repeat(half/(2*a), 2)
    return q, w, edges


def perturb(u, d, delta, clip=True):
    u = np.asarray(u, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if clip:
        return d*(np.clip(u+delta, 0., 1.)-u)
    return d*np.broadcast_to(delta, np.broadcast_shapes(u.shape, delta.shape))


def dense_oracle(u, d, a, clip=True):
    u = np.asarray(u, dtype=np.float64)
    if u.size > 8:
        raise ValueError("dense oracle is forbidden above eight tokens")
    q, w, edges = nodes_weights(u, a, clip)
    values = perturb(u[None, :], d, q[:, None], clip)
    mu = w @ values
    factor = (values-mu).T * np.sqrt(w)[None, :]
    cov = factor @ factor.T
    return dict(q=q, w=w, edges=edges, values=values, mu=mu, cov=cov, factor=factor)


def polynomial_oracle(u, d, a, clip=True):
    """Independent exact integration using affine coefficients at interval midpoints."""
    u = np.asarray(u, dtype=np.float64)
    if u.size > 8:
        raise ValueError("small oracle only")
    if a == 0:
        return np.zeros_like(u), np.zeros((u.size, u.size)), []
    points = [-a, a]
    if clip:
        for t in u:
            for b in (-float(t), 1-float(t)):
                if -a < b < a:
                    points.append(b)
    points = sorted(set(points))
    terms = []
    mu = np.zeros_like(u)
    for left, right in zip(points[:-1], points[1:]):
        mid, half = (left+right)/2, (right-left)/2
        slope, value = np.empty_like(u), np.empty_like(u)
        for t in range(u.size):
            if clip and u[t]+mid < 0:
                slope[t], value[t] = 0., -d*u[t]
            elif clip and u[t]+mid > 1:
                slope[t], value[t] = 0., d*(1-u[t])
            else:
                slope[t], value[t] = d, d*mid
        mass = (right-left)/(2*a)
        mu += mass*value
        terms.append((mass, half, value, slope))
    cov = np.zeros((u.size, u.size))
    for mass, half, value, slope in terms:
        center = value-mu
        cov += mass*(np.outer(center, center) + half**2/3*np.outer(slope, slope))
    return mu, cov, terms


def polynomial_cubic(terms, mu, indices):
    result = 0.
    for mass, half, value, slope in terms:
        coeff = np.array([1.])
        for i in indices:
            coeff = np.polynomial.polynomial.polymul(coeff, [value[i]-mu[i], slope[i]*half])
        result += mass*sum(coeff[k]/(k+1) for k in range(0, len(coeff), 2))
    return float(result)


@dataclass
class StructuredChannel:
    """Implicit exact Gaussian factor using prefix sums; all stored arrays are 1-D."""
    u: np.ndarray
    d: float
    a: float
    clip: bool = True

    def __post_init__(self):
        self.u = np.array(self.u, dtype=np.float64, copy=True)
        self.q, self.w, _ = nodes_weights(self.u, self.a, self.clip)
        self.sqrtw = np.sqrt(self.w)
        self.low = np.searchsorted(self.q, -self.u, side="left")
        self.high = np.searchsorted(self.q, 1-self.u, side="left")
        self.mu = self.uncentered_action(self.w)

    def uncentered_action(self, p):
        p = np.asarray(p, dtype=np.float64)
        if p.shape != self.q.shape:
            raise ValueError("factor coefficient size mismatch")
        if not self.clip:
            return np.full(self.u.size, self.d*np.dot(self.q, p))
        prefix = np.concatenate(([0.], np.cumsum(p)))
        weighted = np.concatenate(([0.], np.cumsum(p*self.q)))
        return self.d*(-self.u*prefix[self.low]
                       + weighted[self.high]-weighted[self.low]
                       + (1-self.u)*(prefix[-1]-prefix[self.high]))

    def centered_action(self, z):
        p = self.sqrtw*np.asarray(z, dtype=np.float64)
        return self.uncentered_action(p)-self.mu*np.sum(p)

    def sample(self, rng):
        # Deliberately no clamp or renormalization after this addition.
        return self.mu+self.centered_action(rng.standard_normal(self.q.size))

    @property
    def array_bytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))


def independent_marginals(u, d, a, clip=True):
    """Exact scalar moments for E*. No token-by-token covariance matrix."""
    u = np.asarray(u, dtype=np.float64)
    if a == 0:
        return np.zeros_like(u), np.zeros_like(u)
    if not clip:
        return np.zeros_like(u), np.full_like(u, d*d*a*a/3)
    # Six nodes per coordinate, independent of T. Zero-length intervals have zero weight.
    edges = np.stack((np.full_like(u, -a), np.clip(-u, -a, a),
                      np.clip(1-u, -a, a), np.full_like(u, a)), axis=1)
    mid, half = (edges[:, :-1]+edges[:, 1:])/2, np.diff(edges, axis=1)/2
    q = np.stack((mid-half/np.sqrt(3.), mid+half/np.sqrt(3.)), axis=2).reshape(len(u), 6)
    w = np.repeat(half/(2*a), 2, axis=1)
    g = perturb(u[:, None], d, q)
    mu = np.sum(w*g, axis=1)
    var = np.sum(w*(g-mu[:, None])**2, axis=1)
    return mu, var


def householder(mu, anchor, degeneracy_tol=1e-12):
    mu = np.asarray(mu, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    ones_unit = np.ones(mu.size)/np.sqrt(mu.size)
    mu_center = mu-ones_unit*np.dot(ones_unit, mu)
    v = anchor-ones_unit*np.dot(ones_unit, anchor)
    if np.linalg.norm(mu_center) > degeneracy_tol:
        unit = mu_center/np.linalg.norm(mu_center)
        v -= unit*np.dot(unit, v)
    norm = float(np.linalg.norm(v))
    if norm <= degeneracy_tol:
        return None, dict(status="invalid_degenerate_complement", projected_anchor_norm=norm)
    v /= norm
    return np.eye(mu.size)-2*np.outer(v, v), dict(status="candidate", projected_anchor_norm=norm, v=v.tolist())


def independent_torch_adapter(tokens_with_cls, delta, clip=True):
    """Fresh implementation with explicitly supplied shifts; CPU only, no RNG emulation."""
    import torch
    if tokens_with_cls.device.type != "cpu" or delta.device.type != "cpu":
        raise ValueError("fixture is CPU-only")
    patches = tokens_with_cls[:, 1:, :]
    lo = patches.amin(dim=1, keepdim=True)
    span = patches.amax(dim=1, keepdim=True)-lo+1e-8
    normalized = (patches-lo)/span+delta
    if clip:
        normalized = normalized.clamp(0., 1.)
    augmented = normalized*span+lo
    return torch.cat((tokens_with_cls[:, :1, :], augmented), dim=1)
