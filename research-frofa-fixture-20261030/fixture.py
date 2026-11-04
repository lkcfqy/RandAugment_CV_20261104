"""Prespecified exact identities and small synthetic checks. No training."""
import ast
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import tracemalloc

START = time.monotonic()
ROOT = Path(os.environ.get('FIXTURE_ROOT', Path(__file__).resolve().parent))
P = json.loads((ROOT/'protocol.json').read_text())
OUT = ROOT/'results'
OUT.mkdir(exist_ok=True)

import numpy as np
import torch
from controls import (normalize_channel, nodes_weights, perturb, dense_oracle,
                      polynomial_oracle, polynomial_cubic, StructuredChannel,
                      independent_marginals, householder, independent_torch_adapter)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
np.seterr(all='raise')
CHECKS = []
DETAILS = {}
ATOL = P['tolerances']['identity_atol']
RTOL = P['tolerances']['identity_rtol']


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=serial, ensure_ascii=False)+'\n')


def check(name, passed, **details):
    record = dict(name=name, passed=bool(passed), details=details)
    CHECKS.append(record)
    with (OUT/'checks.jsonl').open('a') as f:
        f.write(json.dumps(record, default=serial)+'\n')
    print(('PASS ' if passed else 'FAIL ')+name, flush=True)


def close(name, actual, expected, scale=1., atol=ATOL, rtol=RTOL):
    a, b = np.asarray(actual, dtype=np.float64)/scale, np.asarray(expected, dtype=np.float64)/scale
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    check(name, a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol),
          normalized_max_abs_error=error, normalization_scale=scale, atol=atol, rtol=rtol)


def source_checks():
    pins = {'feature_aug.py.txt': '5680796db88811dfb79b6c6291ff4806114118acc31ea62534adc629ad5b22f9',
            'Network_vit.py.txt': 'a1e8dfbbf6d011ba1522ad3e64dd64cd72ed5daf5246e49df50dd164da2bf584'}
    anchors = {'feature_aug.py.txt': {139: 'def brightness_augmentation_v3',
               143: 'torch.min(x, dim=1, keepdim=True)', 144: 'torch.max(x, dim=1, keepdim=True)',
               146: 'denorm = x_max_val - x_min_val + 1e-8',
               150: 'torch.empty(x_norm.shape[0], 1, x_norm.shape[2])',
               155: 'x_norm = torch.clamp', 157: 'x_aug = x_norm * denorm + x_min_val'},
               'Network_vit.py.txt': {221: 'def augment(self, x, delta)',
               222: 'cls_token = x[:, 0, :]', 223: 'patch_token = x[:, 1:, :]',
               224: 'brightness_augmentation_v3(patch_token, max_delta=delta)',
               225: 'torch.cat((cls_token.unsqueeze(1), aug_patch_token), dim=1)'}}
    evidence = []
    for name, sha in pins.items():
        data = (ROOT/'source'/name).read_bytes()
        lines = data.decode().splitlines()
        check('source_hash/'+name, hashlib.sha256(data).hexdigest() == sha)
        # AST parsing is read-only; the tree is never compiled or executed.
        ast.parse(data.decode())
        for line, text in anchors[name].items():
            check(f'source_anchor/{name}:{line}', text in lines[line-1])
        evidence.append(dict(path='source/'+name, sha256=sha,
                             git_blob_sha1=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),
                             anchors={str(k): lines[k-1] for k in anchors[name]}, executed=False))
    save('source-map.json', evidence)


def small_exact_checks():
    rng = np.random.default_rng(P['seed'])
    rows = []
    for case, x in P['small_x_cases'].items():
        u, d, lo = normalize_channel(x, P['epsilon'])
        for a in P['amplitudes']:
            name = f'exact/{case}/a={a}'
            oracle = dense_oracle(u, d, a)
            pmu, pcov, terms = polynomial_oracle(u, d, a)
            control = StructuredChannel(u, d, a)
            emu, evar = independent_marginals(u, d, a)
            close(name+'/quadrature_mass', np.sum(oracle['w']), 1.)
            check(name+'/positive_weights_sorted_nodes',
                  np.all(oracle['w'] > 0) and np.all(np.diff(oracle['q']) >= 0))
            close(name+'/polynomial_mean', oracle['mu'], pmu, d)
            close(name+'/polynomial_full_cov', oracle['cov'], pcov, d*d)
            close(name+'/structured_mean', control.mu, oracle['mu'], d)
            # Every column is checked: this tests the full exact linear map, not MC covariance.
            implicit = np.column_stack([control.centered_action(z) for z in np.eye(control.q.size)])
            close(name+'/all_implicit_factor_columns', implicit, oracle['factor'], d)
            close(name+'/implicit_full_cov', implicit@implicit.T, oracle['cov'], d*d)
            close(name+'/E_scalar_mean', emu, oracle['mu'], d)
            close(name+'/E_scalar_variance', evar, np.diag(oracle['cov']), d*d)
            z = rng.standard_normal(control.q.size)
            close(name+'/fixed_normal_draw_action', control.mu+control.centered_action(z),
                  oracle['mu']+oracle['factor']@z, d)
            shifts = np.unique(np.r_[[-a, 0., a], -u, 1-u])
            shifts = shifts[(shifts >= -a) & (shifts <= a)]
            delta = perturb(u[None, :], d, shifts[:, None])
            close(name+'/endpoint_reconstruction', np.asarray(x)[None, :]+delta,
                  lo+d*np.clip(u[None, :]+shifts[:, None], 0., 1.), max(1., np.max(np.abs(x))))
            check(name+'/bounded_original_C', np.all(delta/d >= -u-ATOL) and np.all(delta/d <= 1-u+ATOL))
            if a == 0:
                close(name+'/zero_shift', oracle['mu'], np.zeros_like(u), d)
                close(name+'/zero_cov', oracle['cov'], np.zeros_like(oracle['cov']), d*d)
            # Repeated coordinates must have identical factor rows (full shared-channel law).
            for i in range(len(u)):
                for j in range(i):
                    if u[i] == u[j]:
                        close(name+f'/repeat_rows_{i}_{j}', oracle['factor'][i], oracle['factor'][j], d)
            rows.append(dict(case=case, a=a, d=d, u=u, mu=oracle['mu'], cov=oracle['cov'],
                             nodes=control.q.size, bytes=control.array_bytes))
    save('small-exact.json', rows)


def endpoint_and_unclipped():
    for a in P['endpoint_math_oracle']['a']:
        o = dense_oracle(np.array([0., 1.]), 1., a)
        close(f'endpoint_math/{a}/mean', o['mu'], np.array([a/4, -a/4]))
        close(f'endpoint_math/{a}/cov', o['cov'], a*a/48*np.array([[5., 3.], [3., 5.]]))
        eig = np.linalg.eigvalsh(o['cov'])
        close(f'endpoint_math/{a}/eigenvalues', eig, np.array([a*a/24, a*a/6]))
        check(f'endpoint_math/{a}/rank', np.count_nonzero(eig > 1e-14) == (2 if a > 0 else 0))
    rows = []
    for case, x in P['small_x_cases'].items():
        u, d, _ = normalize_channel(x)
        for a in P['amplitudes']:
            o = dense_oracle(u, d, a, clip=False)
            st = StructuredChannel(u, d, a, clip=False)
            close(f'no_clip/{case}/{a}/mean', o['mu'], np.zeros_like(u), d)
            close(f'no_clip/{case}/{a}/cov', o['cov'], np.full((len(u), len(u)), d*d*a*a/3), d*d)
            implicit = np.column_stack([st.centered_action(z) for z in np.eye(st.q.size)])
            close(f'no_clip/{case}/{a}/implicit_factor', implicit, o['factor'], d)
            rows.append(dict(case=case, a=a, mu=o['mu'], covariance=o['cov']))
    # Official epsilon semantics on a constant channel: not silently zeroed.
    u, d, _ = normalize_channel([3., 3., 3., 3.])
    o = dense_oracle(u, d, 0.37)
    close('epsilon/constant_d', d, 1e-8, 1e-8)
    close('epsilon/constant_nonzero_mean', o['mu'], np.full(4, d*0.37/4), d)
    close('epsilon/constant_nonzero_cov', o['cov'], np.full((4, 4), d*d*5*0.37**2/48), d*d)
    check('epsilon/constant_not_forced_zero', np.all(o['mu'] > 0) and np.trace(o['cov']) > 0)
    save('no-clip.json', rows)


def dtype_checks():
    examples = [np.array([[0., 1.], [0.2, 1.000000000002], [1., 1.000000000009]]),
                np.full((3, 2), 3.),
                np.array([[0., 1.], [2e-12, 1.+2e-12], [9e-12, 1.+9e-12]])]
    rows = []
    for dtype, tol in [(torch.float32, P['tolerances']['dtype_float32_scaled_atol']),
                       (torch.float64, P['tolerances']['dtype_float64_scaled_atol'])]:
        for index, patches in enumerate(examples):
            x = torch.tensor(np.r_[np.array([[9., -8.]]), patches][None], dtype=dtype, device='cpu')
            before = x.clone()
            for a in [0., 0.37]:
                delta = torch.tensor([[[a, -a/2]]], dtype=dtype, device='cpu')
                y = independent_torch_adapter(x, delta)
                name = f'dtype/{dtype}/{index}/{a}'
                check(name+'/CPU_no_grad', y.device.type == 'cpu' and not y.requires_grad)
                check(name+'/dtype', y.dtype == dtype)
                check(name+'/CLS_bitwise', torch.equal(y[:, :1], before[:, :1]))
                check(name+'/input_unchanged', torch.equal(x, before))
                x64 = x.numpy().astype(np.float64)
                expected = x64.copy()
                model_delta = []
                for c in range(patches.shape[1]):
                    u, d, low = normalize_channel(x64[0, 1:, c])
                    dc = float(delta[0, 0, c])
                    expected[0, 1:, c] = low+d*np.clip(u+dc, 0., 1.)
                    model_delta.append(perturb(u, d, dc).tolist())
                close(name+'/promoted_oracle_output', y.numpy(), expected,
                      scale=max(1., float(np.max(np.abs(expected)))), atol=tol, rtol=0.)
                rows.append(dict(dtype=str(dtype), index=index, amplitude=a,
                                 input_quantized=x.numpy().tolist(), output=y.numpy().tolist(),
                                 ideal_delta_at_quantized_input=model_delta,
                                 actual_subtracted_delta=(y[:, 1:]-x[:, 1:]).numpy().tolist()))
    save('dtype.json', dict(torch_default_dtype=str(torch.get_default_dtype()),
                           note='Official torch.empty has no dtype argument. Continuous ideal moments do not emulate default-float32 uniform PRNG; output rounding can erase epsilon-scale changes at nonzero offsets.', rows=rows))


def marginal_and_cubic():
    spec = P['non_gaussian']
    u, d, a = np.array(spec['u']), spec['d'], spec['a']
    o = dense_oracle(u, d, a)
    mu, cov, terms = polynomial_oracle(u, d, a)
    emu, evar = independent_marginals(u, d, a)
    close('marginal/C_E_mu', o['mu'], emu)
    close('marginal/C_E_diag_cov', np.diag(o['cov']), evar)
    check('marginal/C_E_offdiag_different', np.linalg.norm(o['cov']-np.diag(evar)) > 1e-6)
    # Exact coupling of each scalar coordinate, over a fixed complete input grid.
    grid = np.linspace(-a, a, 101)
    for t in range(len(u)):
        shared_coordinate = perturb(u[None, :], d, grid[:, None])[:, t]
        independent_coordinate = perturb(np.array([u[t]]), d, grid[:, None])[:, 0]
        close(f'marginal/full_pushforward_coordinate_{t}', shared_coordinate, independent_coordinate)
    center = o['values']-o['mu']
    scalar = float(o['w'] @ (center[:, 0]**3))
    mixed = float(o['w'] @ (center[:, 0]*center[:, 1]*center[:, 2]))
    close('cubic/scalar_analytic', scalar, a**3/32)
    close('cubic/scalar_polynomial', scalar, polynomial_cubic(terms, mu, (0, 0, 0)))
    close('cubic/mixed_polynomial', mixed, polynomial_cubic(terms, mu, (0, 1, 2)))
    check('cubic/scalar_differs_matched_G', abs(scalar) > P['tolerances']['cubic_abs_min'])
    check('cubic/mixed_differs_matched_G', abs(mixed) > P['tolerances']['cubic_abs_min'])
    # Symmetric Gaussian cubature verifies every centered odd cubic is zero.
    points = np.r_[o['factor'].T, -o['factor'].T]*np.sqrt(o['factor'].shape[1])
    close('cubic/G_scalar_zero', np.mean(points[:, 0]**3), 0.)
    close('cubic/G_mixed_zero', np.mean(points[:, 0]*points[:, 1]*points[:, 2]), 0.)
    close('cubic/E_mixed_zero_by_independence', np.prod(emu[:3]-o['mu'][:3]), 0.)
    save('cubic-and-marginal.json', dict(u=u, d=d, a=a, mu=mu, covariance_C=cov,
                                       variance_E=evar, scalar_C=scalar, scalar_E=scalar,
                                       scalar_G_C=0., scalar_G_E=0., mixed_C=mixed,
                                       mixed_E=0., mixed_G_C=0., mixed_G_E=0.,
                                       marginal_proof='Same measurable scalar map, same scalar uniform law; changing coupling does not change the entire marginal distribution.'))


def affine_squared():
    s = P['affine_squared']
    x, W = np.array(s['x_token_channel']), np.array(s['W'])
    b, y, a = np.array(s['b']), np.array(s['y']), s['a']
    flat = x.T.ravel()
    channels = [dense_oracle(*normalize_channel(x[:, c])[:2], a) for c in range(x.shape[1])]
    mu = np.concatenate([o['mu'] for o in channels])
    covariance = np.zeros((flat.size, flat.size))
    J = sum(o['factor'].shape[1] for o in channels)
    B = np.zeros((flat.size, J))
    start = 0
    for c, o in enumerate(channels):
        sl = slice(c*x.shape[0], (c+1)*x.shape[0])
        covariance[sl, sl] = o['cov']
        B[sl, start:start+o['factor'].shape[1]] = o['factor']
        start += o['factor'].shape[1]
    close('affine/full_channel_block_factor', B@B.T, covariance)
    residual = W@(flat+mu)+b-y

    def loss(delta):
        residuals = (flat[None, :]+delta)@W.T+b-y
        return np.sum(residuals**2, axis=1)

    c_points, c_weights = [], []
    for idx in itertools.product(*[range(o['q'].size) for o in channels]):
        c_points.append(np.concatenate([o['values'][j] for o, j in zip(channels, idx)]))
        c_weights.append(np.prod([o['w'][j] for o, j in zip(channels, idx)]))
    c_points, c_weights = np.array(c_points), np.array(c_weights)
    c_exact = float(c_weights @ loss(c_points))
    c_formula = float(residual@residual + np.trace(W@covariance@W.T))
    c_g = float(np.mean(loss(mu+np.r_[B.T, -B.T]*np.sqrt(J))))
    close('affine/C_product_quadrature_vs_formula', c_exact, c_formula)
    close('affine/C_G_expected_squared', c_exact, c_g)
    product_mu = c_weights@c_points
    product_cov = (c_points-product_mu).T @ (c_weights[:, None]*(c_points-product_mu))
    close('affine/independent_channels_mean', product_mu, mu)
    close('affine/independent_channels_full_cov', product_cov, covariance)

    scalar = []
    for c in range(x.shape[1]):
        u, d, _ = normalize_channel(x[:, c])
        for value in u:
            o = dense_oracle([value], d, a)
            scalar.append(o)
    e_points, e_weights = [], []
    for idx in itertools.product(*[range(o['q'].size) for o in scalar]):
        e_points.append([o['values'][j, 0] for o, j in zip(scalar, idx)])
        e_weights.append(np.prod([o['w'][j] for o, j in zip(scalar, idx)]))
    e_points, e_weights = np.array(e_points), np.array(e_weights)
    e_cov = np.diag(np.diag(covariance))
    e_factor = np.diag(np.sqrt(np.diag(covariance)))
    e_exact = float(e_weights@loss(e_points))
    e_formula = float(residual@residual+np.trace(W@e_cov@W.T))
    e_g = float(np.mean(loss(mu+np.r_[e_factor.T, -e_factor.T]*np.sqrt(flat.size))))
    close('affine/E_product_quadrature_vs_formula', e_exact, e_formula)
    close('affine/E_G_expected_squared', e_exact, e_g)
    close('affine/E_independent_full_cov',
          (e_points-mu).T@(e_weights[:, None]*(e_points-mu)), e_cov)
    save('affine-squared.json', dict(C_exact=c_exact, G_C_exact=c_g, C_formula=c_formula,
                                   E_exact=e_exact, G_E_exact=e_g, E_formula=e_formula,
                                   C_nodes=len(c_points), E_nodes=len(e_points),
                                   mu=mu, full_C_cov=covariance, full_E_cov=e_cov,
                                   note='Known affine squared expected-loss identity only; no CE or trajectory claim.'))


def direction_checks():
    s = P['direction']
    x = np.array(s['x'])
    u, d, _ = normalize_channel(x)
    o = dense_oracle(u, d, s['a'])
    Q, status = householder(o['mu'], np.array(s['anchor']), P['tolerances']['direction_degeneracy_norm'])
    check('direction/primary_complement_exists', Q is not None, status=status)
    if Q is None:
        save('direction.json', status)
        return
    original_x = x.copy()
    centered = o['values']-o['mu']
    rotated = o['mu']+centered@Q.T
    r_cov = Q@o['cov']@Q.T
    relative_change = float(np.linalg.norm(r_cov-o['cov'])/np.linalg.norm(o['cov']))
    valid = relative_change > P['tolerances']['direction_relative_covariance_change_min']
    status.update(status='valid_noncommuting' if valid else 'invalid_commuting', relative_covariance_change=relative_change)
    close('direction/Q_orthogonal', Q.T@Q, np.eye(len(u)))
    close('direction/Q_ones', Q@np.ones(len(u)), np.ones(len(u)))
    close('direction/Q_mu', Q@o['mu'], o['mu'], d)
    close('direction/mean', o['w']@rotated, o['mu'], d)
    close('direction/per_draw_full_energy', np.sum(rotated**2, axis=1), np.sum(o['values']**2, axis=1), d*d)
    close('direction/per_draw_token_mean', np.mean(rotated, axis=1), np.mean(o['values'], axis=1), d)
    close('direction/covariance_spectrum', np.linalg.eigvalsh(r_cov), np.linalg.eigvalsh(o['cov']), d*d)
    close('direction/rotated_factor_cov', (Q@o['factor'])@(Q@o['factor']).T, r_cov, d*d)
    check('direction/nontrivial_covariance_change', valid, relative_covariance_change=relative_change)
    check('direction/input_X_unchanged', np.array_equal(x, original_x))
    check('direction/perturbed_features_actually_change', np.linalg.norm((x+rotated)-(x+o['values'])) > 1e-6)
    check('direction/not_a_token_permutation', np.max(np.minimum(np.abs(Q), np.abs(Q-1))) > 1e-3)
    # Use exactly the same primary Q for shared translation negative control.
    shifts = np.array([-s['a'], -0.13, 0., 0.21, s['a']])
    translations = d*shifts[:, None]*np.ones((1, len(u)))
    close('direction/no_clip_per_draw_negative_control', translations@Q.T, translations, d)
    un_cov = d*d*s['a']**2/3*np.ones((len(u), len(u)))
    close('direction/no_clip_covariance_commutes', Q@un_cov@Q.T, un_cov, d*d)
    negative_rows = []
    for label, xx, a, anchor in [('two_tokens', [0., 1.], 0.43, [1., -2.]),
                                ('constant', [2., 2., 2., 2.], 0.43, [1., -2., 3., -4.]),
                                ('zero_amplitude', s['x'], 0., s['anchor'])]:
        uu, dd, _ = normalize_channel(xx)
        oo = dense_oracle(uu, dd, a)
        qq, st = householder(oo['mu'], anchor)
        if qq is not None:
            denom = np.linalg.norm(oo['cov'])
            change = float(np.linalg.norm(qq@oo['cov']@qq.T-oo['cov'])/denom) if denom else 0.
            st.update(status='invalid_commuting' if change <= P['tolerances']['direction_relative_covariance_change_min'] else 'valid_noncommuting', relative_covariance_change=change)
        check('direction/negative_invalid/'+label, st['status'].startswith('invalid'), status=st)
        negative_rows.append(dict(label=label, **st))
    save('direction.json', dict(primary=status, Q=Q, mu=o['mu'], original_cov=o['cov'], rotated_cov=r_cov,
                                invalid_controls=negative_rows, no_clip_status='invalid_commuting_negative_control'))


def monte_carlo_smoke():
    rng = np.random.default_rng(P['seed_mc'])
    n = P['mc']['n']
    s = P['non_gaussian']
    u, d, a = np.array(s['u']), s['d'], s['a']
    o = dense_oracle(u, d, a)
    C = perturb(u[None, :], d, rng.uniform(-a, a, (n, 1)))
    E = perturb(u[None, :], d, rng.uniform(-a, a, (n, len(u))))
    GC = o['mu']+rng.standard_normal((n, o['factor'].shape[1]))@o['factor'].T
    GE = o['mu']+rng.standard_normal((n, len(u)))*np.sqrt(np.diag(o['cov']))
    result = {}
    for name, sample, cov in [('C', C, o['cov']), ('E', E, np.diag(np.diag(o['cov']))),
                              ('G_C', GC, o['cov']), ('G_E', GE, np.diag(np.diag(o['cov'])))]:
        avg = sample.mean(axis=0)
        empiric_cov = (sample-avg).T@(sample-avg)/n
        close('MC/'+name+'/mean', avg, o['mu'], d, atol=P['mc']['normalized_mean_atol'], rtol=0.)
        close('MC/'+name+'/cov', empiric_cov, cov, d*d, atol=P['mc']['normalized_cov_atol'], rtol=0.)
        result[name] = dict(empirical_mean=avg, empirical_cov=empiric_cov)
    for t in range(len(u)):
        values = np.linspace(-a, a, 15)
        cdf_c = np.mean(C[:, t, None] <= values, axis=0)
        cdf_e = np.mean(E[:, t, None] <= values, axis=0)
        close(f'MC/C_E_CDF/{t}', cdf_c, cdf_e, atol=P['mc']['cdf_atol'], rtol=0.)
    # Gaussian support is deliberately not clipped; this guards the actual draw path.
    check('MC/G_C_not_second_clipped', np.any(GC/d < -u) or np.any(GC/d > 1-u))
    check('MC/G_E_not_second_clipped', np.any(GE/d < -u) or np.any(GE/d > 1-u))
    save('mc.json', dict(n=n, seed=P['seed_mc'], results=result, inferential_role='smoke only'))


def shape_check():
    s = P['shape']
    rng = np.random.default_rng(P['seed_shape'])
    tracemalloc.start()
    wall = time.perf_counter()
    x = rng.standard_normal((s['B'], s['T'], s['C'])).astype(np.float64)
    prepared = []
    e_mu, e_var = np.empty_like(x), np.empty_like(x)
    for c in range(s['C']):
        u, d, _ = normalize_channel(x[0, :, c])
        prepared.append(StructuredChannel(u, d, s['a']))
        e_mu[0, :, c], e_var[0, :, c] = independent_marginals(u, d, s['a'])
    prep_seconds = time.perf_counter()-wall
    wall = time.perf_counter()
    g = np.empty_like(x)
    for c, control in enumerate(prepared):
        g[0, :, c] = control.sample(rng)
    sample_seconds = time.perf_counter()-wall
    wall = time.perf_counter()
    g_e = e_mu+np.sqrt(e_var)*rng.standard_normal(x.shape)
    c_draw, e_draw = np.empty_like(x), np.empty_like(x)
    for c, control in enumerate(prepared):
        c_draw[0, :, c] = perturb(control.u, control.d, rng.uniform(-s['a'], s['a']))
        e_draw[0, :, c] = perturb(control.u, control.d, rng.uniform(-s['a'], s['a'], s['T']))
    remaining_arms_seconds = time.perf_counter()-wall
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    persistent_bytes = sum(control.array_bytes for control in prepared)+e_mu.nbytes+e_var.nbytes
    arrays = [v for control in prepared for v in vars(control).values() if isinstance(v, np.ndarray)]
    check('shape/196x768', list(g.shape) == [s['B'], s['T'], s['C']])
    check('shape/finite_float64', np.isfinite(g).all() and g.dtype == np.float64)
    check('shape/all_four_arms', all(v.shape == x.shape and v.dtype == np.float64 and np.isfinite(v).all()
                                    for v in [g, g_e, c_draw, e_draw]))
    check('shape/only_1D_prepared_arrays', all(v.ndim == 1 for v in arrays))
    check('shape/node_bound', max(c.q.size for c in prepared) <= 4*s['T']+2)
    check('shape/persistent_memory_cap', persistent_bytes <= s['persistent_array_bytes_max'], bytes=persistent_bytes)
    check('shape/traced_peak_cap', peak <= s['tracemalloc_peak_bytes_max'], bytes=peak)
    save('shape-memory.json', dict(shape=list(g.shape), dtype=str(g.dtype),
                                   preprocess_seconds=prep_seconds, one_draw_seconds=sample_seconds,
                                   other_three_arms_seconds=remaining_arms_seconds,
                                   persistent_array_bytes=persistent_bytes, traced_current_bytes=current,
                                   traced_peak_bytes=peak, process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   maxrss_units='bytes on macOS; includes Python, Torch imports and earlier checks',
                                   naive_token_covariance_bytes=s['T']**2*s['C']*8,
                                   nodes_min=min(c.q.size for c in prepared), nodes_max=max(c.q.size for c in prepared),
                                   output_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                                   output_sha256_by_arm={name: hashlib.sha256(v.tobytes()).hexdigest()
                                                         for name, v in [('C', c_draw), ('E', e_draw), ('G_C', g), ('G_E', g_e)]},
                                   one_pass_only=True, batching_note='B=1: per-sample loops generalize algebraically; no throughput claim for larger batches.'))
    np.save(OUT/'shape-input.npy', x, allow_pickle=False)
    np.save(OUT/'shape-one-gaussian-draw.npy', g, allow_pickle=False)
    np.savez(OUT/'shape-other-arms.npz', C=c_draw, E=e_draw, G_E=g_e)


def main():
    environment = dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                       torch=torch.__version__, platform=platform.platform(), machine=platform.machine(),
                       device='cpu', oracle_dtype='float64', torch_threads=torch.get_num_threads(),
                       grad_enabled=torch.is_grad_enabled(), started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                       downloads_during_fixture=0, training_steps=0, third_party_source_executed=False)
    save('environment.json', environment)
    groups = [source_checks, small_exact_checks, endpoint_and_unclipped, dtype_checks,
              marginal_and_cubic, affine_squared, direction_checks, monte_carlo_smoke, shape_check]
    timings = {}
    for fn in groups:
        before = time.monotonic()
        try:
            fn()
        except Exception:
            check(fn.__name__+'/exception', False, traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        timings[fn.__name__] = time.monotonic()-before
    failed = [v for v in CHECKS if not v['passed']]
    summary = dict(passed=len(CHECKS)-len(failed), failed=len(failed), total=len(CHECKS),
                   failed_names=[v['name'] for v in failed], timings_seconds=timings,
                   worker_wall_seconds=time.monotonic()-START,
                   control_qualified=not failed,
                   interpretation='Only qualified implementation controls if all required checks pass; no research contribution or training go-ahead.',
                   completed_utc=datetime.datetime.now(datetime.UTC).isoformat())
    save('summary.json', summary)
    save('checks.json', CHECKS)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
