"""Pre-scan calibration; no downloaded Python code is executed."""
import time
import traceback
import cv2
import numpy as np
from audit_core import *


def main():
    guard('fixture', 180)
    started = time.time()
    save_json('environment.json', environment())
    (ROOT / 'opencv-build.txt').write_text(cv2.getBuildInformation())
    report = dict(start_utc=utc(), fixture_corrections=0, tests=[])
    tests = report['tests']
    try:
        for shape in [(7, 9), (512, 512), (2064, 2494)]:
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            maps = dict(x=x, y=y, map_x=x.astype(np.float32), map_y=y.astype(np.float32))
            xx, yy = recover_donors(maps)
            assert np.array_equal(xx, x) and np.array_equal(yy, y)
            check_uint8(maps, xx, yy)
            tests.append(dict(name='identity_and_uint8_coordinate_encoding', shape=shape, pass_=True))
            del maps, x, y, xx, yy

        x, y = np.meshgrid(np.arange(17), np.arange(9))
        for offset in [-0.51, -0.49, 0.49, 0.51]:
            mx = np.clip(x + offset, 0, 16).astype(np.float32)
            my = np.clip(y + offset, 0, 8).astype(np.float32)
            maps = dict(x=x, y=y, map_x=mx, map_y=my)
            xx, yy = recover_donors(maps)
            assert np.array_equal(xx, np.floor(mx + 0.5))
            assert np.array_equal(yy, np.floor(my + 0.5))
            check_uint8(maps, xx, yy)
            tests.append(dict(name='signed_fraction_and_edge_corner_clip', offset=offset, pass_=True))

        vals, kinds = [], []
        for center in [0.5, 1.5, 2.5, 7.5, 15.5]:
            f = np.float32(center)
            for kind, value in [('below', np.nextafter(f, np.float32(-np.inf))),
                                ('tie', f), ('above', np.nextafter(f, np.float32(np.inf)))]:
                vals.append(value); kinds.append(kind)
        mx = np.array([vals], dtype=np.float32)
        my = np.full_like(mx, 4)
        donor = cv2.remap(x.astype(np.float32), mx, my, cv2.INTER_NEAREST)[0]
        ties = []
        for v, kind, d in zip(vals, kinds, donor):
            lo, hi = int(np.floor(v)), int(np.ceil(v))
            assert int(d) in [lo, hi]
            if kind == 'below': assert int(d) == lo
            if kind == 'above': assert int(d) == hi
            ties.append(dict(value=float(v), kind=kind, donor=int(d), nearest_even=int(np.rint(v))))
        tests.append(dict(name='half_ties_and_adjacent_float32', observations=ties, pass_=True))

        kernels = {}
        impulse = np.zeros((257, 257), np.float64); impulse[128, 128] = 1
        border_probe = np.random.RandomState(20261030).rand(97, 103)
        for sigma in [3, 5, 10]:
            k = 8 * sigma + 1
            automatic = cv2.GaussianBlur(impulse, (0, 0), sigma)
            explicit = cv2.GaussianBlur(impulse, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)
            assert np.array_equal(automatic, explicit)
            support = np.flatnonzero(automatic[128] > 0)
            assert len(support) == k
            default_border = cv2.GaussianBlur(border_probe, (0, 0), sigma)
            reflect = cv2.GaussianBlur(border_probe, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
            assert np.array_equal(default_border, reflect)
            paper_kernel = cv2.GaussianBlur(impulse, (6*sigma+1, 6*sigma+1), sigma)
            weights = cv2.getGaussianKernel(k, sigma).ravel()
            def sq_effective(pos):
                acc = {}
                for offset, weight in zip(range(-k//2+1, k//2+1), weights):
                    idx = cv2.borderInterpolate(pos + offset, 512, cv2.BORDER_REFLECT_101)
                    acc[idx] = acc.get(idx, 0.0) + float(weight)
                return sum(v*v for v in acc.values())
            interior = sq_effective(k)
            corner = sq_effective(0)
            kernels[str(sigma)] = dict(kernel=k, radius=k//2, support_first=int(support[0]),
                support_last=int(support[-1]), auto_explicit_exact=True, default_reflect101_exact=True,
                paper_kernel=6*sigma+1, auto_paper_max_abs=float(np.max(np.abs(automatic-paper_kernel))),
                one_dim_interior_weight_sq=interior, one_dim_boundary_weight_sq=corner,
                alpha1_interior_variance=interior**2/3,
                alpha1_corner_variance=corner**2/3, corner_variance_multiplier=(corner/interior)**2)
        report['kernels'] = kernels
        tests.append(dict(name='auto_kernel_and_default_border', pass_=True))

        # One calibration seed for positive control (also present in the fixed scan).
        local = np.random.RandomState(0)
        maps = make_maps((512, 512), [(100, 3)], local)
        xx, yy = recover_donors(maps)
        changed = (xx != maps['x']) | (yy != maps['y'])
        assert np.count_nonzero(changed) > 0, 'positive control failed'
        check_uint8(maps, xx, yy)
        tests.append(dict(name='positive_control_100_3_seed0', changed=int(changed.sum()), pass_=True))
        report['positive_control_pass'] = True
        for prob in [0, 1]:
            np.random.seed(991)
            before = state_hash(np.random.get_state())
            value, applied = gate(prob)
            after = state_hash(np.random.get_state())
            manual = np.random.RandomState(991); manual.rand()
            assert before != after and after == state_hash(manual.get_state())
            assert applied == bool(prob)
            tests.append(dict(name='gate_consumes_global_draw', prob=prob, value=value,
                              applied=applied, before=before, after=after, pass_=True))
        report['status'] = 'pass'
    except Exception as exc:
        report['status'] = 'fail'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report.update(elapsed_seconds=time.time()-started, peak_rss_bytes=peak_rss(), end_utc=utc())
        save_json('fixture-results.json', report)
        checkpoint('fixture_finished', status=report['status'], peak_rss_bytes=peak_rss(),
                   elapsed_seconds=report['elapsed_seconds'])


if __name__ == '__main__':
    main()
