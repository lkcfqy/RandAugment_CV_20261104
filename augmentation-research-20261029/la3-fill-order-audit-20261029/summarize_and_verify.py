#!/usr/bin/env python3
"""Post-run descriptive summaries and independently executed sampler replay checks.

Does not alter the frozen protocol, original audit, policy files, or raw cases.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path
import random
import zipfile
import numpy as np
from PIL import Image
import audit

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def main():
    policies = audit.read_policies()
    pairs = {p['pair_id']: p for p in json.loads((RESULTS/'swap-candidates.json').read_text())}
    rows = [json.loads(line) for line in (RESULTS/'all-cases.jsonl').read_text().splitlines()]
    fail = [json.loads(line) for line in (RESULTS/'failure-cases.jsonl').read_text().splitlines()]
    assert len(rows) == 3584 and len({r['case_id'] for r in rows}) == len(rows)
    assert {r['case_id'] for r in fail} == {r['case_id'] for r in rows if not r['strict_fill_only']}
    with zipfile.ZipFile(RESULTS/'failure-arrays.zip') as archive:
        assert set(archive.namelist()) == {r['case_id']+'.npz' for r in fail}
        assert archive.testzip() is None
    groups = {}
    for label, chosen in [('strict_primary', [r for r in rows if r['strict_primary_candidate']]),
                          ('other_chain_controls', [r for r in rows if not r['strict_primary_candidate']])]:
        groups[label] = {
            'n': len(chosen), 'status':dict(collections.Counter(r['status'] for r in chosen)),
            'source_max_abs':max(r['source_max_abs'] for r in chosen),
            'tolerance_one_pass':sum(r['tolerance_one_fill_only'] for r in chosen),
            'severity_l2_matches_all':sum(r['full_image_l2_exact_match'] for r in chosen),
            'severity_l2_matches_nonzero_strict':sum(r['strict_fill_only'] and r['full_image_l2_exact_match'] for r in chosen),
            'severity_by_canvas':{c:{'strict_nonzero_n':sum(r['strict_fill_only'] for r in chosen if r['canvas']==c),
                                    'strict_nonzero_l2_match':sum(r['strict_fill_only'] and r['full_image_l2_exact_match'] for r in chosen if r['canvas']==c)}
                                  for c in sorted({r['canvas'] for r in chosen})}}
    conditional = {}
    for dataset, by_class in policies.items():
        entries=[]; by_geometry=collections.defaultdict(list)
        for cls, triples in sorted(by_class.items()):
            for index,t in enumerate(triples):
                if not audit.strict_candidate(t): continue
                g=next(op for op in t if op in audit.GEO)
                r={'class_id':cls,'entry_index':index,'triple':t,'geometry':g,'invert_after_geometry':int(t.index('Invert')>t.index(g)),'entry_probability_given_class':1/len(triples)}
                entries.append(r); by_geometry[g].append(r)
        strata={}
        varying_entries=0
        for g, items in sorted(by_geometry.items()):
            classes={}
            for cls in sorted({r['class_id'] for r in items}):
                local=[r for r in items if r['class_id']==cls]
                n_after=sum(r['invert_after_geometry'] for r in local)
                classes[cls]={'n':len(local),'n_after':n_after,'q_after':n_after/len(local),'both_orders_native':0<n_after<len(local)}
            variation=len({c['q_after'] for c in classes.values()})>1
            if variation: varying_entries+=len(items)
            strata[g]={'classes':classes,'cross_class_shared_geometry':len(classes)>1,'q_varies_between_classes':variation}
        total=sum(map(len,by_class.values()))
        conditional[dataset]={'n_classes':len(by_class),'total_entries':total,'strict_entries':len(entries),'equal_class_strict_event_mass':len(entries)/total,'varying_q_entries':varying_entries,'equal_class_varying_q_event_mass':varying_entries/total,'entries':entries,'strata':strata,
                             'scope':'Z fixes geometric operator, signed magnitude and dimensions; identical native magnitude law follows from code. Real image content and post-crop joint support are unverified. Entry mass is not empirical training incidence.'}
    # Independent execution of the original class sampler, with original RNGs,
    # compared to the parameter-tape replay used by the audit.
    ns=audit.load_functions()
    tree=ast.parse((ROOT/'sources/code/LA3/data.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PDA_Aug_Policies_Op')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original-PDA_Aug_Policies_Op','exec'),ns)
    replay=[]
    canvas=audit.canvases()['bounded_ramp']
    for dataset, by_class in policies.items():
        classes=sorted({r['class_id'] for r in conditional[dataset]['entries']})
        for cls in classes:
            for case in range(12):
                seed=20261029+case
                np.random.seed(seed); random.seed(seed)
                direct=np.asarray(ns['PDA_Aug_Policies_Op'](by_class[cls])(Image.fromarray(canvas)))
                selected=int(np.random.RandomState(seed).choice(range(len(by_class[cls])),1)[0])
                triple=by_class[cls][selected]
                tape=audit.actions_for(triple,case,(32,32))
                played=audit.run_chain(canvas,tape,[0,1,2])[0]
                maximum=int(np.abs(direct.astype(np.int16)-played.astype(np.int16)).max())
                replay.append({'dataset':dataset,'class_id':cls,'seed':seed,'entry_index':selected,'triple':triple,'max_abs_error':maximum})
    assert all(r['max_abs_error']==0 for r in replay)
    # Every geometry-source atom is retained in noncommuting-chain failures.
    failures_by_third={}
    for op in ['AutoContrast','Equalize','Contrast','Sharpness','Cutout','ShearX','ShearY','TranslateX','TranslateY','Rotate']:
        selected=[r for r in rows if not r['strict_primary_candidate'] and op in pairs[r['pair_id']]['triple']]
        failures_by_third[op]={'n':len(selected),'status':dict(collections.Counter(r['status'] for r in selected)),
                             'first_nonfill_failure':next((r['case_id'] for r in selected if r['status']=='nonfill_changed'),None)}
    frozen=json.loads((ROOT/'protocol-freeze.json').read_text())
    summary={'groups':groups,'failure_archive_entries':len(fail),'sampler_replay_checks':len(replay),'sampler_replay_max_abs':max(r['max_abs_error'] for r in replay),'failures_by_included_operation_overlapping_groups':failures_by_third,'interpretation':'descriptive fixed synthetic cases, no confidence intervals or empirical native effect estimate'}
    (RESULTS/'conditional-support.json').write_text(json.dumps(conditional,indent=2)+'\n')
    (RESULTS/'sampler-replay-checks.json').write_text(json.dumps(replay,indent=2)+'\n')
    (RESULTS/'descriptive-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
