"""Independent numerical/analytic checks. All scientific cases are retained."""
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import audit


def validate(cases,windows,precision,protocol):
    results=[]
    def check(name,condition,detail):
        results.append({'name':name,'passed':bool(condition),'detail':detail})
    small=D('1e-65')
    all_rows=[r for c in cases for s in c['samples'] for r in s['steps']]
    check('finite_CE_independent_loss_expression',
          all(abs(r['ce_signed_difference']-r['ce_logpartition_difference'])<small for r in all_rows),
          'Log-partition increment agrees with direct CE(z_new)-CE(z_old).')
    check('both_KL_directions_independent_probability_definition',
          all(abs(r[k]-r[k+'_direct'])<small for r in all_rows for k in ('kl_new_old','kl_old_new')),
          'Natural-parameter KL agrees with direct sum of two probability-log-ratio terms.')
    check('KL_nonnegative',all(r[k]>=-small for r in all_rows for k in ('kl_new_old','kl_old_new')),
          'Decimal reference, without clipping or substituting negative values.')
    check('global_CE_Taylor_remainder_bound',
          all(abs(r['ce_signed_difference']-r['gradient_previous_signed_dot'])<=r['delta_logit']**2/8+small for r in all_rows),
          'Logistic CE second derivative is <=1/4, so absolute first-order remainder <=delta^2/8.')
    check('global_KL_quadratic_bound',
          all(r[k]<=r['delta_logit']**2/8+small for r in all_rows for k in ('kl_new_old','kl_old_new')),
          'Both Bernoulli KL directions are bounded by delta-logit squared /8.')
    errors=[]
    h=D('1e-20')
    for p in (D('.1'),D('.5'),D('.9')):
        z=audit.logit(p)
        for y in (0,1):
            numeric=(audit.ce(z+h,y)-audit.ce(z-h,y))/(2*h)
            errors.append(abs(numeric-(p-y)))
    check('CE_gradient_central_finite_difference',max(errors)<D('1e-38'),
          'Independent central differences at p=.1,.5,.9 and both labels; h=1e-20.')
    check('coherent_shared_trajectory',
          all(s['steps'][i]['z_current']==s['steps'][i+1]['z_previous']
              for c in cases for s in c['samples'] for i in range(len(s['steps'])-1)),
          'No restarting probabilities between the two/five/ten updates.')
    gd_errors=[]
    for c in cases:
        if c['spec']['kind']!='actual_GD':continue
        for u in c['optimizer_updates']:
            def objective(w):
                return sum((audit.ce(s['offset']+s['slope']*w,s['y']) for s in c['samples']),D(0))
            w=u['w_previous']
            numerical_grad=(objective(w+h)-objective(w-h))/(2*h)
            gd_errors.append(abs((u['w_current']-w)/u['eta']+numerical_grad))
    check('actual_GD_step_vs_objective_finite_difference',max(gd_errors)<D('1e-38'),
          'Actual-GD steps checked against an independently finite-differenced full two-sample SUM loss.')
    check('90_vs_60_digit_precision',
          max(precision['decimal_60_vs_90_max_relative_error'].values())<D('1e-35'),
          'All nonzero reference step metrics agree to at least 35 relative decimal digits.')
    base={(w['case_id'],w['window_length'],w['end_step']):w for w in windows if w['common_eta_divisor']=='1'}
    scaled=[]
    for w in windows:
        if w['kind']!='prescribed':continue
        ref=base[(w['case_id'],w['window_length'],w['end_step'])]
        eta=D(w['common_eta_divisor'])
        scaled.extend(abs(v*eta*eta-v0)<small for m in audit.METRICS for v,v0 in zip(w['ssd'][m],ref['ssd'][m]))
        scaled.extend(w['relations'][m]==ref['relations'][m] for m in audit.METRICS)
    check('identical_constant_eta_normalization',all(scaled),
          'All metrics divided by same eta. Window SSD scales by 1/eta^2 and rankings remain unchanged; not a claim for changing trajectories/schedules.')
    primary=[w for w in windows if w['case_id'].startswith('primary_positive_two') and w['common_eta_divisor']=='1']
    check('analytic_small_step_reversal',
          all(w['comparisons'][m+'_vs_'+k]=='reversal' for w in primary for m in audit.METRICS[:3] for k in audit.METRICS[3:]),
          'Predicted by (1-p) coefficient vs p(1-p) coefficient; checks exact finite trajectories at every declared epsilon, not supplied reviewer numbers.')
    concord=[w for w in windows if w['case_id'].startswith(('concordant_confident','different_movement'))]
    check('concordant_controls',
          all(w['comparisons']['ce_abs_difference_vs_kl_new_old']=='concordant' for w in concord),
          'Includes same movement with p=.5,.9 and equal initial p with different movement magnitudes.')
    labels=[c for c in cases if c['spec']['id']=='same_prediction_different_label']
    check('label_blindness_control',
          all(a['kl_new_old']==b['kl_new_old'] and a['ce_abs_difference']!=b['ce_abs_difference']
              for c in labels for a,b in zip(c['samples'][0]['steps'],c['samples'][1]['steps'])),
          'Identical probability trajectories, different fixed true labels: KL identical, supervised loss response different.')
    stationary=[c for c in cases if c['spec']['id']=='stationary']
    check('stationary_controls',all(r[m]==0 for c in stationary for s in c['samples'] for r in s['steps'] for m in audit.METRICS),
          'Zero parameter movement gives exactly zero scores in every definition.')
    returning=[w for w in windows if w['case_id'].startswith('mixed_direction_five') and w['window_length']==2 and w['end_step'] in (2,4)]
    check('out_and_back_absolute_CE_variance_exactly_zero',
          all(w['ssd']['ce_abs_difference']==[D(0),D(0)] for w in returning),
          'Exact path reversal gives equal absolute CE differences; residual numerical ordering is not scientific evidence.')
    tiny=[w for w in primary if w['case_id'].endswith('_eps_0.00000001')][0]
    eps=D('1e-8')
    check('analytic_SSD_leading_coefficients',
          all(abs(v/(eps**2)-(1-p)**2/2)<D('1e-7') for p,v in zip((D('.1'),D('.5')),tiny['ssd']['grad_previous_abs_dot'])) and
          all(abs(v/(eps**4)-D(9)/8*p*p*(1-p)*(1-p))<D('1e-7') for p,v in zip((D('.1'),D('.5')),tiny['ssd']['kl_new_old'])),
          'Two increments epsilon,2epsilon: gradient SSD coefficient=(1-p)^2/2; KL SSD coefficient=9[p(1-p)]^2/8.')
    manifest=json.loads((audit.ROOT/'sources.json').read_text())
    check('pinned_source_hashes',all(hashlib.sha256((audit.ROOT/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['records']),
          'All saved arXiv-v1 source bytes agree with retrieval manifest.')
    return {'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),'checks':results}
