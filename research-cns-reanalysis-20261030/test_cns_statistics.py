"""SYNTHETIC NONEMPIRICAL checks for weights, paired contrasts and resource decisions."""
import unittest
import numpy as np
from cns_core import MODELS, SHIFTS, assemble
from cns_statistics import (make_design, statistic_vector, STAT_NAMES, source_weights,
                            standardized_probability, bootstrap, resource_gate, summaries)
from test_cns_core import make_rows


def synthetic_design(all_correct=False):
    by_model={}
    for m, model in enumerate(MODELS):
        rows=[]
        for c in (2,4):
            for seed in range(1,7):
                for s, shift in enumerate(SHIFTS):
                    y=[True]*6
                    if not all_correct:
                        if seed%2==0 and s==0: y=[True,True,True,False,True,False]
                        if seed%3==0 and s==1: y=[True,False,True,False,False,False]
                        if seed==1 and m==1 and s==1: y=[False,True,True,True,True,True]
                    rows+=make_rows(y,shift,class_id=c,seed=seed)
        by_model[model]=assemble(rows,model)[0]
    return make_design(by_model)


class StatisticsTests(unittest.TestCase):
    def test_source_strata_sizes_are_fixed(self):
        sources=((1,1),(1,2),(2,1),(2,2),(2,3))
        w=source_weights(sources,np.random.default_rng(9))
        self.assertEqual(w[:2].sum(),2)
        self.assertEqual(w[2:].sum(),3)

    def test_class_cluster_preserves_all_records_in_class(self):
        sources=((1,1),(1,2),(2,1),(2,2),(2,3))
        w=source_weights(sources,np.random.default_rng(4),'class_cluster')
        self.assertEqual(w[0],w[1]); self.assertTrue(np.all(w[2:]==w[2]))

    def test_contrasts_are_paired_and_endpoint_cohort_is_fixed(self):
        d=synthetic_design(); vec=statistic_vector(d,np.ones(len(d.sources)))
        for m in range(2):
            v=vec[m*len(STAT_NAMES):(m+1)*len(STAT_NAMES)]
            self.assertAlmostEqual(v[3],v[1]-v[0])
            self.assertAlmostEqual(v[4],v[2]-v[1])
        self.assertEqual(d.common[0].sum(),12)
        self.assertEqual(d.common[1].sum(),10)
        self.assertEqual(d.cross_model_common.sum(),10)

    def test_joint_replicate_changes_match_direct_probability_changes(self):
        d=synthetic_design(); w=source_weights(d.sources,np.random.default_rng(3))
        v=statistic_vector(d,w)
        for m in range(2):
            offset=m*len(STAT_NAMES)
            self.assertAlmostEqual(v[offset+4],v[offset+2]-v[offset+1])

    def test_empty_fixed_class_is_not_silently_dropped(self):
        p=standardized_probability(np.array([True,False]),np.array([True,False]),
                                   np.ones(2),np.array([1,2]),(1,2))
        self.assertTrue(np.isnan(p))

    def test_recovery_final_scale_is_na_not_zero_evidence(self):
        s=summaries(synthetic_design())
        last=[r for r in s['recovery_by_first_error'] if r['first_error_scale']==2.5]
        self.assertTrue(all(r['recovery_rate'] is None for r in last))

    def test_reproducible_joint_bootstrap(self):
        d=synthetic_design()
        a=bootstrap(d,replicates=40,seed=11); b=bootstrap(d,replicates=40,seed=11)
        self.assertEqual(a['simultaneous_primary_intervals'],b['simultaneous_primary_intervals'])

    def test_zero_event_bootstrap_flagged_not_proof_of_absence(self):
        b=bootstrap(synthetic_design(True),replicates=30)
        self.assertTrue(b['degenerate_bootstrap'])
        self.assertIn('degenerate',resource_gate(b,identity_verified=True)['decision'])

    def test_negative_large_effect_not_mislabeled_stable(self):
        b={'valid_fraction':1.,'degenerate_bootstrap':False,
           'simultaneous_primary_intervals':[[-.2,-.1]]*4}
        self.assertEqual(resource_gate(b,True)['decision'],'candidate_for_independent_semantic_check_only')

    def test_gate_requires_entire_interval_inside_threshold(self):
        b={'valid_fraction':1.,'degenerate_bootstrap':False,
           'simultaneous_primary_intervals':[[-.01,.02]]*4}
        self.assertTrue(resource_gate(b,True)['decision'].startswith('close_candidate'))
        b['simultaneous_primary_intervals'][0]=[-.06,.01]
        self.assertIn('crosses',resource_gate(b,True)['decision'])

    def test_identity_missing_means_undetermined_not_negative(self):
        b=bootstrap(synthetic_design(),replicates=20)
        self.assertIn('identity_not_verified',resource_gate(b)['decision'])
        self.assertEqual(resource_gate(b,True,synthetic=True)['decision'],'not_applicable_synthetic_nonempirical')


if __name__=='__main__': unittest.main(verbosity=2)
