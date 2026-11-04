"""Read-only result integrity checks; does not repeat coordinate experiments."""
import csv
import hashlib
import json
from datetime import datetime
import numpy as np
from audit_core import ROOT, PROTOCOL, save_json, utc


def main():
    checks = []
    with (ROOT/'per-seed.csv').open() as f: rows = list(csv.DictReader(f))
    expected = {(h,w,seed,a,s) for h,w in PROTOCOL['stage_a']['shapes']
                for seed in range(128) for a,s in PROTOCOL['stage_a']['parameters']}
    keys = [(int(r['height']),int(r['width']),int(r['seed']),int(r['alpha']),int(r['sigma'])) for r in rows]
    assert len(rows) == len(set(keys)) == 1024 and set(keys) == expected
    checks.append('exact preregistered 1024 rows; all seed/shape/parameter tuples; no duplicates')
    region_names = ['interior','edge','corner_tl','corner_tr','corner_bl','corner_br']
    for row in rows:
        n, changed = int(row['pixels']), int(row['changed_count'])
        assert 0 <= changed <= n
        assert abs(float(row['change_rate'])-changed/n) < 1e-15
        assert (row['applied_identity'] == 'True') == (changed == 0)
        assert row['applied'] == 'True' and row['skipped'] == 'False'
        assert row['global_before'] != row['global_after_gate'] == row['global_after_transform']
        assert sum(int(row[k+'_n']) for k in region_names) == n
        assert sum(int(row[k+'_changed']) for k in region_names) == changed
        for name in region_names:
            assert abs(float(row[name+'_rate']) - int(row[name+'_changed'])/int(row[name+'_n'])) < 1e-15
        if row['alpha'] == '1':
            assert changed == 0 and int(row['donor_linf']) == 0 and float(row['map32_linf']) < .5
        else: assert changed > 0 and int(row['donor_linf']) > 0
        assert int(row['peak_rss_bytes']) < 2**31
    assert sum(r['uint8_encoding_verified'] == 'True' for r in rows) == 8
    checks.append('CSV rates, applied-vs-skipped, RNG-state relationships, disjoint regions, controls and uint8 checks consistent')
    sources = json.loads((ROOT/'sources/manifest.json').read_text())
    for row in sources:
        assert 'error' not in row
        data = (ROOT/'sources'/row['file']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    assert (ROOT/'sources/transforms-head.py').read_bytes() == (ROOT/'sources/transforms-addition.py').read_bytes()
    checks.append('all downloaded source/HTML/license bytes match recorded SHA256; two commit files identical')
    fixture = json.loads((ROOT/'fixture-results.json').read_text())
    scan = json.loads((ROOT/'scan-summary.json').read_text())
    trace = json.loads((ROOT/'trace-summary.json').read_text())
    assert fixture['status'] == 'pass' and fixture['fixture_corrections'] == 0
    assert scan['status'] == trace['status'] == 'complete'
    assert scan['elapsed_seconds'] <= 1800 and trace['elapsed_seconds'] <= 600
    assert max(datetime.fromisoformat(x['end_utc']).timestamp() for x in [fixture,scan,trace]) < PROTOCOL['deadline_epoch']
    if (ROOT/'budget.json').exists():
        assert json.loads((ROOT/'budget.json').read_text())['accounting_epoch'] < PROTOCOL['deadline_epoch']
    assert max(fixture['peak_rss_bytes'],scan['peak_rss_bytes'],trace['peak_rss_bytes']) < 2**31
    checks.append('fixture passed without correction; scan/trace complete within time and memory budgets')
    with np.load(ROOT/'trace-outputs-A.npz', allow_pickle=False) as a, \
         np.load(ROOT/'trace-outputs-B.npz', allow_pickle=False) as b, \
         np.load(ROOT/'trace-outputs-C.npz', allow_pickle=False) as c:
        for key in ['image','mask','sample_order']:
            assert np.array_equal(b[key], c[key])
        assert np.array_equal(a['sample_order'], b['sample_order'])
        for key in ['image','mask']:
            assert sum(np.array_equal(x,y) for x,y in zip(a[key],b[key])) == 0
    checks.append('reloaded saved arrays: B=C exactly, A differs on every final image/mask, sample orders equal')
    records = json.loads((ROOT/'trace-records.json').read_text())
    states = json.loads((ROOT/'trace-states.json').read_text())
    for b,c in zip(states['B'], states['C']):
        assert b == c
    for variant in ['B','C']:
        assert all(r['applied_identity'] and not r['mask_shares_input'] for r in records[variant])
    checks.append('complete saved B/C local and global RNG states equal; both outputs are new arrays')
    save_json('validation.json', dict(status='pass', utc=utc(), checks=checks, rows=len(rows),
                                    fixtures=12, fixture_corrections=0))
    print(json.dumps(dict(status='pass', checks=len(checks), rows=len(rows))))


if __name__ == '__main__': main()
