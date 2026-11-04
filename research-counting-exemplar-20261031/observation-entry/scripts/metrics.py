"""Frozen paired-count error and pair-level oracle gates. Pure standard library."""
from fractions import Fraction
from numbers import Integral

MODES = ('F', 'C', 'V')


def image_error(n_A, n_B, n_hat_A, n_hat_B):
    values = (n_A, n_B, n_hat_A, n_hat_B)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError('truth and predicted counts must be nonnegative integers')
    denominator = int(n_A)+int(n_B)
    if denominator <= 0:
        raise ValueError('official per-image A+B total must be positive')
    return Fraction(abs(int(n_hat_A)-int(n_A))+abs(int(n_hat_B)-int(n_B)), denominator)


def summarize(pairs, policy):
    """Every eligible pair must contain its exact two images and all six queries/image.

    policy is the frozen roster including NA rows. This function never selects pairs,
    changes names, drops difficult images, clips large errors, or assesses localization.
    """
    if len({p['pair_id'] for p in policy}) != len(policy):
        raise ValueError('duplicate policy pair')
    expected = {p['pair_id'] for p in policy if p['eligible']}
    if not expected:
        raise ValueError('no eligible pairs')
    if len({p['pair_id'] for p in pairs}) != len(pairs) or {p['pair_id'] for p in pairs} != expected:
        raise ValueError('prediction pairs must match the full eligible roster exactly')
    actual = {p['pair_id']:p for p in pairs}
    output = []; differences = []; qualifying = 0
    def exact(value): return [value.numerator, value.denominator]
    for entry in policy:
        pair_id = entry['pair_id']
        if not entry['eligible']:
            if not entry.get('reason'):
                raise ValueError('NA requires an explicit reason')
            output.append({'pair_id':pair_id,'status':'NA','reason':entry['reason'],'mean_error':None,'mean_error_exact':None,'qualifies':None})
            continue
        rows = actual[pair_id]['images']
        image_ids = entry['image_ids']
        if len(image_ids) != 2 or len(set(image_ids)) != 2 or len(rows) != 2 or {im['image_id'] for im in rows} != set(image_ids):
            raise ValueError('each pair requires its exact two distinct fixed images')
        errors = {mode:[] for mode in MODES}
        image_output = []
        for im in rows:
            try:
                truth = im['truth']; predictions = im['predictions']
                if set(truth) != {'A','B'} or set(predictions) != set(MODES):
                    raise ValueError('both targets and all three modes required')
                row_errors = {}
                for mode in MODES:
                    if set(predictions[mode]) != {'A','B'}:
                        raise ValueError('A/B predictions required for every image/mode')
                    error = image_error(truth['A'],truth['B'],predictions[mode]['A'],predictions[mode]['B'])
                    errors[mode].append(error); row_errors[mode] = exact(error)
                image_output.append({'image_id':im['image_id'],'error_exact':row_errors})
            except KeyError as exc:
                raise ValueError('incomplete image/mode/target prediction') from exc
        means = {mode:sum(errors[mode],Fraction(0))/2 for mode in MODES}
        oracle = min(means['C'],means['V'])
        difference = oracle-means['F']
        condition = means['F'] <= Fraction(1,10) and means['C'] > Fraction(3,10) and means['V'] > Fraction(3,10)
        differences.append(difference); qualifying += int(condition)
        output.append({'pair_id':pair_id,'status':'EVALUATED','images':image_output,
                       'mean_error':{k:float(v) for k,v in means.items()},'mean_error_exact':{k:exact(v) for k,v in means.items()},
                       'oracle_min_C_V':float(oracle),'oracle_minus_F_exact':exact(difference),'qualifies':condition})
    G = sum(differences,Fraction(0))/len(expected)
    gate_G = G >= Fraction(1,10)
    gate_pairs = qualifying >= 10
    return {'pairs':output,'eligible_pairs':len(expected),'NA_pairs':len(policy)-len(expected),
            'G':float(G),'G_exact':exact(G),'qualifying_pairs':qualifying,
            'gate_G':gate_G,'gate_pair_count':gate_pairs,'numeric_investment_gate':gate_G and gate_pairs,
            'oracle_envelope_uses_ground_truth':True,'deployable_selector':False,
            'target_selection_interpretation':'UNASSESSED_REQUIRES_PREDICTED_BOX_REVIEW',
            'post_effect_removal_allowed':False,'placeholder_points_used':False}
