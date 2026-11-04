"""Joint source-key bootstrap with fixed membership; empirical use needs identity evidence."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from cns_core import MODELS, SHIFTS, SCALES, paired_sources


@dataclass
class Design:
    sources: tuple
    present: np.ndarray  # model, shift, source
    correct: np.ndarray  # model, shift, source, scale
    first: np.ndarray
    terminal: np.ndarray
    common: np.ndarray  # model, source; same eligibility for both shifts
    cross_model_common: np.ndarray
    fixed_classes: tuple
    audits: dict


def make_design(trajectories_by_model: dict, verified_sources: set | None = None) -> Design:
    sources = tuple(sorted(set().union(*({key for _, key in tr} for tr in trajectories_by_model.values()))))
    lookup = {key: i for i, key in enumerate(sources)}
    n = len(sources)
    present = np.zeros((2, 2, n), dtype=bool)
    correct = np.zeros((2, 2, n, 6), dtype=bool)
    first = np.full((2, 2, n), np.nan)
    terminal = np.full_like(first, np.nan)
    common = np.zeros((2, n), dtype=bool)
    audits = {}
    class_support = []
    for m, model in enumerate(MODELS):
        trajectories = trajectories_by_model[model]
        _, eligible, audits[model] = paired_sources(trajectories)
        for key in eligible:
            if verified_sources is None or key in verified_sources:
                common[m, lookup[key]] = True
        for s, shift in enumerate(SHIFTS):
            class_support.append({key[0] for sh, key in trajectories if sh == shift})
            for (sh, key), trajectory in trajectories.items():
                if sh != shift:
                    continue
                i = lookup[key]
                f = trajectory.features
                present[m, s, i] = True
                correct[m, s, i] = trajectory.sequence
                if f['first_error'] is not None:
                    first[m, s, i] = f['first_error']
                if f['terminal_error'] is not None:
                    terminal[m, s, i] = f['terminal_error']
    fixed_classes = tuple(sorted(set.intersection(*class_support)))
    return Design(sources, present, correct, first, terminal, common,
                  common.all(axis=0), fixed_classes, audits)


def probability(event, mask, weights):
    denominator = np.dot(mask, weights)
    return float(np.dot(mask & event, weights) / denominator) if denominator > 0 else np.nan


def standardized_probability(event, mask, weights, class_ids, fixed_classes):
    if not fixed_classes:
        return np.nan
    class_ids=np.asarray(class_ids,dtype=int)
    support=np.asarray(fixed_classes,dtype=int)
    length=max(int(class_ids.max(initial=0)),int(support.max()))+1
    denominator=np.bincount(class_ids,weights=weights*mask,minlength=length)[support]
    numerator=np.bincount(class_ids,weights=weights*(mask & event),minlength=length)[support]
    # Do not silently drop unsupported eligible classes; bincount avoids K full scans.
    return float(np.mean(numerator/denominator)) if np.all(denominator>0) else np.nan


STAT_NAMES = ('delta_F_publisher', 'delta_F_common', 'delta_T_common', 'D_C', 'D_E',
              'delta_F_publisher_class_equal', 'delta_F_common_class_equal', 'delta_T_common_class_equal',
              'D_C_class_equal', 'D_E_class_equal', 'delta_F_cross_model_common',
              'delta_T_cross_model_common', 'D_E_cross_model_common')


def statistic_vector(design: Design, weights) -> np.ndarray:
    values = []
    class_ids = np.array([key[0] for key in design.sources])
    for m in range(2):
        released = [probability(design.first[m, s] == 1.5, design.present[m, s], weights) for s in range(2)]
        common_f = [probability(design.first[m, s] == 1.5, design.common[m], weights) for s in range(2)]
        common_t = [probability(design.terminal[m, s] == 1.5, design.common[m], weights) for s in range(2)]
        a, b, c = released[0]-released[1], common_f[0]-common_f[1], common_t[0]-common_t[1]
        values.extend((a, b, c, b-a, c-b))
        standard = []
        for events, masks in ((design.first[m],design.present[m]),
                              (design.first[m],np.broadcast_to(design.common[m], (2,len(weights)))),
                              (design.terminal[m],np.broadcast_to(design.common[m], (2,len(weights))))):
            p = [standardized_probability(events[s]==1.5,masks[s],weights,class_ids,design.fixed_classes) for s in range(2)]
            standard.append(p[0]-p[1])
        a, b, c = standard
        values.extend((a,b,c,b-a,c-b))
        f = [probability(design.first[m,s]==1.5,design.cross_model_common,weights) for s in range(2)]
        t = [probability(design.terminal[m,s]==1.5,design.cross_model_common,weights) for s in range(2)]
        df, dt = f[0]-f[1], t[0]-t[1]
        values.extend((df,dt,dt-df))
    return np.asarray(values)


def source_weights(sources, rng, mode='stratified_source'):
    """One multiplicity per whole source: all shifts/scales/models travel together."""
    class_ids = np.array([key[0] for key in sources])
    classes = np.unique(class_ids)
    weights = np.zeros(len(sources), dtype=float)
    if mode == 'stratified_source':
        for c in classes:
            ids = np.flatnonzero(class_ids == c)
            weights[ids] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)))
    elif mode == 'class_cluster':
        multiplicities = rng.multinomial(len(classes), np.full(len(classes), 1/len(classes)))
        for c, multiplicity in zip(classes, multiplicities):
            weights[class_ids == c] = multiplicity
    else:
        raise ValueError('Unknown bootstrap mode')
    return weights


def bootstrap(design: Design, replicates=4000, seed=20261030, mode='stratified_source'):
    if not design.sources:
        raise ValueError('No sources')
    point = statistic_vector(design, np.ones(len(design.sources)))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(STAT_NAMES)*2))
    for b in range(replicates):
        draws[b] = statistic_vector(design, source_weights(design.sources, rng, mode))
    coordinates = np.array([3,4,len(STAT_NAMES)+3,len(STAT_NAMES)+4])
    valid = np.isfinite(draws[:,coordinates]).all(axis=1)
    draws_valid = draws[valid]
    result = {'mode':mode, 'replicates':replicates, 'valid_replicates':int(valid.sum()),
              'valid_fraction':float(valid.mean()), 'seed':seed, 'point':point.tolist(),
              'coordinate_order':[f'{m}/{name}' for m in MODELS for name in STAT_NAMES],
              'primary_coordinate_indices':coordinates.tolist(), 'percentile_intervals':None,
              'simultaneous_primary_intervals':None, 'simultaneous_half_width':None}
    result['valid_replicates_per_coordinate'] = np.isfinite(draws).sum(axis=0).tolist()
    if len(draws_valid) and np.isfinite(point[coordinates]).all():
        intervals = np.full((len(point),2), np.nan)
        for j in range(len(point)):
            selected = draws_valid[:,j]
            selected = selected[np.isfinite(selected)]
            if len(selected) >= .99*replicates and np.isfinite(point[j]):
                intervals[j] = np.quantile(selected,[.025,.975])
        q = float(np.quantile(np.abs(draws_valid[:,coordinates]-point[coordinates]).max(axis=1), .95))
        result.update(percentile_intervals=intervals.tolist(), simultaneous_half_width=q,
                      simultaneous_primary_intervals=np.column_stack((point[coordinates]-q, point[coordinates]+q)).tolist(),
                      degenerate_bootstrap=(q==0),
                      zero_width_percentile_coordinates=np.flatnonzero(intervals[:,0]==intervals[:,1]).tolist())
    return result


def resource_gate(boot, identity_verified=False, synthetic=False, threshold=.05):
    if synthetic:
        return {'decision':'not_applicable_synthetic_nonempirical'}
    if not identity_verified:
        return {'decision':'undetermined_source_identity_not_verified'}
    if boot['valid_fraction'] < .99 or boot.get('simultaneous_primary_intervals') is None:
        return {'decision':'undetermined_bootstrap_support'}
    if boot.get('degenerate_bootstrap', False):
        return {'decision':'undetermined_degenerate_bootstrap_not_proof_of_no_events'}
    bounds = np.asarray(boot['simultaneous_primary_intervals'])
    inside = (bounds[:,0] > -threshold) & (bounds[:,1] < threshold)
    outside = (bounds[:,1] <= -threshold) | (bounds[:,0] >= threshold)
    if inside.all():
        decision='close_candidate_within_prespecified_resource_threshold'
    elif np.all(inside | outside) and outside.any():
        decision='candidate_for_independent_semantic_check_only'
    else:
        decision='undetermined_interval_crosses_resource_threshold'
    return {'decision':decision,'threshold':threshold,'units':'trajectory proportion, not accuracy'}


def summaries(design: Design):
    weights = np.ones(len(design.sources))
    class_ids = np.array([key[0] for key in design.sources])
    proportions, accuracies, recovery, composition, support = [], [], [], [], []
    for m, model in enumerate(MODELS):
        both = design.present[m].all(axis=0)
        for s, shift in enumerate(SHIFTS):
            cohorts = {'publisher_complete':design.present[m,s], 'common_all_firstpoint_states':both,
                       'common_first_observed_correct':design.common[m],
                       'cross_model_common_first_observed_correct':design.cross_model_common}
            for cohort, mask in cohorts.items():
                n = int(mask.sum())
                empty = [int(c) for c in design.fixed_classes if not (mask & (class_ids==c)).any()]
                support.append({'model':model,'shift':shift,'cohort':cohort,'n':n,
                                'class_equal_supported':bool(design.fixed_classes) and not empty,
                                'empty_fixed_classes':empty,
                                'first_observed_errors':int((mask & ~design.correct[m,s,:,0]).sum())})
                for c in np.unique(class_ids):
                    composition.append({'model':model,'shift':shift,'cohort':cohort,'class_id':int(c),
                                        'n':int((mask & (class_ids==c)).sum())})
                for endpoint, events in (('F',design.first[m,s]),('T',design.terminal[m,s])):
                    for scale in SCALES:
                        event = events == scale
                        count = int((mask & event).sum())
                        proportions.append({'model':model,'shift':shift,'cohort':cohort,'endpoint':endpoint,
                                            'scale':scale,'events':count,'n':n,
                                            'proportion':probability(event,mask,weights),
                                            'class_equal_proportion':standardized_probability(event,mask,weights,class_ids,design.fixed_classes),
                                            'zero_events_not_proof_of_absence':count==0})
                for i, scale in enumerate(SCALES):
                    accuracies.append({'model':model,'shift':shift,'cohort':cohort,'scale':scale,'n':n,
                                       'accuracy':probability(design.correct[m,s,:,i],mask,weights)})
                    stratum = mask & (design.first[m,s]==scale)
                    denominator = int(stratum.sum())
                    count = int((stratum & design.correct[m,s,:,i+1:].any(axis=1)).sum())
                    recovery.append({'model':model,'shift':shift,'cohort':cohort,'first_error_scale':scale,
                                     'recovered':count,'first_errors':denominator,'later_observations':5-i,
                                     'recovery_rate':count/denominator if denominator and i<5 else None,
                                     'interpretation':'no_later_observation' if i==5 else 'descriptive_only',
                                     'small_stratum':denominator<20})
    return {'event_proportions':proportions,'accuracy':accuracies,'recovery_by_first_error':recovery,
            'class_composition':composition,'cohort_support':support,'fixed_class_support_K':list(design.fixed_classes)}
