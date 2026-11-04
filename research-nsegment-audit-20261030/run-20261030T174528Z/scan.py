"""Fixed 128-seed/two-size scan. All actual donors come from cv2.remap."""
import csv
import json
import time
import traceback
import cv2
import numpy as np
from audit_core import *


def component_metrics(row, axis, field, integer, mapping, upper):
    row['preclip_linf_' + axis] = float(np.max(np.abs(field)))
    row['preclip_rms_' + axis] = float(np.sqrt(np.mean(field * field)))
    row['map32_linf_' + axis] = float(np.max(np.abs(mapping - integer)))
    preclip = integer + field
    clipped = (preclip < 0) | (preclip > upper)
    row['clip_' + axis + '_count'] = int(np.count_nonzero(clipped))
    row['clip_' + axis + '_ratio'] = row['clip_' + axis + '_count'] / mapping.size
    del preclip
    distance = np.abs(mapping - np.floor(mapping) - np.float32(0.5))
    ties = distance == 0
    near = (distance > 0) & (distance <= np.spacing(mapping))
    row['half_tie_' + axis + '_count'] = int(np.count_nonzero(ties))
    row['near_tie_' + axis + '_count'] = int(np.count_nonzero(near))
    return clipped


def main():
    fixture = json.loads((ROOT / 'fixture-results.json').read_text())
    assert fixture['status'] == 'pass' and fixture['positive_control_pass']
    started = time.time()
    deadline = guard('coordinate_scan', 1800)
    checkpoint('scan_started', expected_rows=1024, deadline_epoch=deadline,
               order='shape, seed, fixed parameter list', fixture_corrections=fixture['fixture_corrections'])
    summary = dict(status='running', start_utc=utc(), expected_rows=1024, rows=0, groups={},
                   target_counterexample=None, peak_rss_bytes=0)
    writer = None
    try:
        with (ROOT / 'per-seed.csv').open('x', newline='') as f:
            for shape in PROTOCOL['stage_a']['shapes']:
                h, w = shape
                for seed in PROTOCOL['stage_a']['seeds']:
                    for alpha, sigma in PROTOCOL['stage_a']['parameters']:
                        t0 = time.time()
                        np.random.seed(seed)
                        global_before = state_hash(np.random.get_state())
                        value, applied = gate(1)
                        global_after_gate = state_hash(np.random.get_state())
                        assert applied
                        maps = make_maps((h, w), [(alpha, sigma)], np.random.RandomState(seed))
                        global_after_transform = state_hash(np.random.get_state())
                        assert global_after_gate == global_after_transform
                        donor_x, donor_y = recover_donors(maps)
                        changed = (donor_x != maps['x']) | (donor_y != maps['y'])
                        count = int(np.count_nonzero(changed))
                        k = fixture['kernels'][str(sigma)]['kernel']
                        row = dict(height=h, width=w, seed=seed, alpha=alpha, sigma=sigma, prob=1,
                                   gate_value=value, applied=True, skipped=False,
                                   applied_identity=(count == 0), pixels=h*w, changed_count=count,
                                   change_rate=count/(h*w), kernel=k, radius=k//2,
                                   global_before=global_before, global_after_gate=global_after_gate,
                                   global_after_transform=global_after_transform,
                                   local_before=maps['local_before'],
                                   local_after_choice=maps['local_after_choice'],
                                   local_after_field=maps['local_after_field'])
                        cx = component_metrics(row, 'x', maps['dx'], maps['x'], maps['map_x'], w-1)
                        cy = component_metrics(row, 'y', maps['dy'], maps['y'], maps['map_y'], h-1)
                        row['clip_either_count'] = int(np.count_nonzero(cx | cy))
                        row['clip_either_ratio'] = row['clip_either_count'] / (h*w)
                        del cx, cy
                        row['donor_linf_x'] = int(np.max(np.abs(donor_x - maps['x'])))
                        row['donor_linf_y'] = int(np.max(np.abs(donor_y - maps['y'])))
                        row['donor_linf'] = max(row['donor_linf_x'], row['donor_linf_y'])
                        row['map32_linf'] = max(row['map32_linf_x'], row['map32_linf_y'])
                        row.update(region_counts(changed, k//2))
                        row['uint8_encoding_verified'] = False
                        if seed == 0 or (alpha == 1 and sigma == 10 and count):
                            check_uint8(maps, donor_x, donor_y)
                            row['uint8_encoding_verified'] = True
                        row['elapsed_seconds'] = time.time()-t0
                        row['peak_rss_bytes'] = peak_rss()
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); f.flush()
                        summary['rows'] += 1
                        key = f'{h}x{w}/a{alpha}_s{sigma}'
                        group = summary['groups'].setdefault(key, dict(seeds=0, identity_seeds=0,
                            changed_pixels=0, pixels=0, min_rate=1., max_rate=0.,
                            max_map32_linf=0., max_preclip_linf=0., max_donor_linf=0,
                            min_preclip_rms_x=1e9, max_preclip_rms_x=0., first_nonzero_seed=None))
                        group['seeds'] += 1
                        group['identity_seeds'] += int(count == 0)
                        group['changed_pixels'] += count; group['pixels'] += h*w
                        group['min_rate'] = min(group['min_rate'], row['change_rate'])
                        group['max_rate'] = max(group['max_rate'], row['change_rate'])
                        group['max_map32_linf'] = max(group['max_map32_linf'], row['map32_linf'])
                        group['max_preclip_linf'] = max(group['max_preclip_linf'], row['preclip_linf_x'], row['preclip_linf_y'])
                        group['max_donor_linf'] = max(group['max_donor_linf'], row['donor_linf'])
                        group['min_preclip_rms_x'] = min(group['min_preclip_rms_x'], row['preclip_rms_x'])
                        group['max_preclip_rms_x'] = max(group['max_preclip_rms_x'], row['preclip_rms_x'])
                        if count and group['first_nonzero_seed'] is None: group['first_nonzero_seed'] = seed
                        summary['peak_rss_bytes'] = peak_rss()
                        if alpha == 100 and count == 0:
                            raise AssertionError('positive control failed during fixed scan; fixture invalid')
                        if alpha == 1 and sigma == 10 and count:
                            yy, xx = np.nonzero(changed); yy, xx = yy[:1024], xx[:1024]
                            np.savez_compressed(ROOT / 'target-counterexample.npz', seed=seed, shape=shape,
                                y=yy, x=xx, donor_x=donor_x[yy,xx], donor_y=donor_y[yy,xx],
                                map_x=maps['map_x'][yy,xx], map_y=maps['map_y'][yy,xx],
                                dx=maps['dx'][yy,xx], dy=maps['dy'][yy,xx])
                            summary['target_counterexample'] = dict(shape=shape, seed=seed,
                                changed_count=count, uint8_verified=True, file='target-counterexample.npz')
                            summary['status'] = 'stopped_target_nonzero'
                            return
                        del maps, donor_x, donor_y, changed
                    if seed % 16 == 15 or seed == 0:
                        summary['elapsed_seconds'] = time.time()-started
                        save_json('scan-summary.json', summary)
                        checkpoint('scan_progress', shape=shape, seed_completed=seed, rows=summary['rows'],
                                   elapsed_seconds=summary['elapsed_seconds'], peak_rss_bytes=peak_rss())
            summary['status'] = 'complete'
    except Exception as exc:
        summary['status'] = 'error'
        summary['error'] = str(exc)
        summary['traceback'] = traceback.format_exc()
        raise
    finally:
        summary.update(elapsed_seconds=time.time()-started, end_utc=utc(), peak_rss_bytes=peak_rss())
        save_json('scan-summary.json', summary)
        checkpoint('scan_finished', status=summary['status'], rows=summary['rows'],
                   elapsed_seconds=summary['elapsed_seconds'], peak_rss_bytes=peak_rss())


if __name__ == '__main__':
    main()
