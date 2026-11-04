"""Conservative promotion guards implementing protocol lines21,23,37.

Only existing simultaneous intervals are used. No scores, tuning choices,
thresholds, temperatures, or confidence-family dimensions are changed.
"""
import math

METHODS=('aps','raps','saps','rankonly')

def finite(x):
    try:return x is not None and math.isfinite(float(x))
    except (TypeError,ValueError):return False

def tuning_eligibility(param_feasible,selection):
    parameters=all(bool(param_feasible.get(m,False)) for m in METHODS)
    temperatures=all(bool(selection[m]['sources'][source]['feasible']) for m in METHODS for source in ('p','q'))
    return {'parameters_feasible':parameters,'temperatures_feasible':temperatures,
            'all_feasible':parameters and temperatures}

def assess_control(control_point,q_point,intervals,control_key,q_key,tuning_feasible):
    out={'point_no_larger_and_comparable':False,'in_primary_simultaneous_family':False,
         'point_comparable':False,'simultaneous_comparability_established':False,
         'mechanism_beyond_control_established':False,'tuning_feasible':bool(tuning_feasible),
         'comparison_status':'UNASSESSED'}
    if not all(finite(p.get(k)) for p in (control_point,q_point) for k in ('M','coverage','Dhat')):
        out['comparison_status']='UNESTIMABLE_POINT_METRIC';return out
    rp,qp=control_point,q_point
    comparable=rp['coverage']>=.89 and qp['coverage']>=.89 and abs(rp['coverage']-qp['coverage'])<=.01 and rp['Dhat']<=qp['Dhat']+.001
    out['point_comparable']=bool(comparable)
    out['point_no_larger_and_comparable']=bool(comparable and rp['M']<=qp['M'])
    required=[f'{k}:{metric}' for k in (control_key,q_key) for metric in ('M','coverage','Dhat')]
    available=all(k in intervals for k in required)
    out['in_primary_simultaneous_family']=available
    if not tuning_feasible:out['comparison_status']='TUNING_INFEASIBLE';return out
    if not available:out['comparison_status']='OUTSIDE_FIXED_CI_FAMILY';return out
    for key in required:
        it=intervals[key];bounds=it.get('simultaneous95')
        if not it.get('estimable',False) or not isinstance(bounds,(list,tuple)) or len(bounds)!=2 or not all(finite(v) for v in bounds):
            out['comparison_status']='UNESTIMABLE_INTERVAL';return out
    def b(key,metric):return intervals[f'{key}:{metric}']['simultaneous95']
    rc,qc=b(control_key,'coverage'),b(q_key,'coverage')
    rd,qd=b(control_key,'Dhat'),b(q_key,'Dhat')
    rm,qm=b(control_key,'M'),b(q_key,'M')
    cdiff=[rc[0]-qc[1],rc[1]-qc[0]]
    ddiff_upper=rd[1]-qd[0]
    ci_comparable=rc[0]>=.89 and qc[0]>=.89 and cdiff[0]>=-.01 and cdiff[1]<=.01 and ddiff_upper<=.001
    out.update(coverage_difference_bounds_from_existing_CI=cdiff,Ddiff_upper_from_existing_CI=ddiff_upper,
               size_superiority_lower_from_existing_CI=rm[0]-qm[1],
               simultaneous_comparability_established=bool(ci_comparable))
    out['mechanism_beyond_control_established']=bool(comparable and ci_comparable and rm[0]-qm[1]>.25)
    if not comparable:out['comparison_status']='INCOMPARABLE_POINT'
    elif not ci_comparable:out['comparison_status']='COMPARABILITY_UNCERTAIN'
    elif not out['mechanism_beyond_control_established']:out['comparison_status']='SIZE_SUPERIORITY_NOT_ESTABLISHED'
    else:out['comparison_status']='CONTROL_EXCLUDED_AT_PROTOCOL_MARGIN'
    return out
