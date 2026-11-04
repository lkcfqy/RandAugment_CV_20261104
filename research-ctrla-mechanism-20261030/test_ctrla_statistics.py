"""Synthetic arithmetic/cluster fixtures, not experimental evidence."""
import unittest
import numpy as np
from frozen_diagnostics import mechanism_metrics,compare_renderers,audit_matching,bootstrap_weights

def output(errors,ce=None):
    errors=np.asarray(errors,dtype=int)
    if errors.ndim==1:errors=errors[:,None]
    return {'prediction':errors,'target':np.zeros_like(errors),'ce':np.asarray(ce) if ce is not None else np.ones_like(errors,dtype=float)}

class StatisticsTests(unittest.TestCase):
    def test_paired_sign_and_cluster_retention(self):
        n=2000;w=bootstrap_weights(n)
        s=np.zeros((n,15),int);s[:200]=1;f=np.zeros_like(s)
        r=compare_renderers(output(s),output(s),output(f),w)['S_minus_F']['aggregate']
        self.assertAlmostEqual(r['estimate'],.1);self.assertLess(r['ci95'][0],.1);self.assertGreater(r['ci95'][1],.1)
        # Every image's 15 views agree, so uncertainty is image Bernoulli uncertainty.
        self.assertGreater(r['ci95'][1]-r['ci95'][0],.02)
    def test_interaction_and_events(self):
        n=2000;w=bootstrap_weights(n);zero=np.zeros((n,15),int);full=zero.copy();full[:100]=1
        r=mechanism_metrics(output(np.zeros(n)),output(zero),output(zero),output(full),w)
        self.assertAlmostEqual(r['aggregate']['E']['estimate'],.05)
        self.assertAlmostEqual(r['aggregate']['new_wrong']['estimate'],.05)
        self.assertEqual(r['aggregate']['corrected_wrong']['estimate'],0.)
        self.assertAlmostEqual(r['by_operation'][0]['conditional_R']['estimate'],.95)
    def test_invalid_low_denominator(self):
        n=2000;w=bootstrap_weights(n);one=np.ones((n,15),int);zero=np.zeros_like(one)
        r=mechanism_metrics(output(np.zeros(n)),output(zero),output(one),output(one),w)
        self.assertFalse(r['aggregate']['all_ROR_strata_valid'])
        self.assertIsNone(r['by_operation'][0]['conditional_R'])
        self.assertEqual(r['aggregate']['E']['estimate'],0.)
    def test_CE_ratio_of_means_and_acceptance(self):
        n=2000;w=bootstrap_weights(n);ref=output(np.zeros((n,15)),np.ones((n,15)))
        exact=output(np.zeros((n,15)),np.full((n,15),1.02));bad=output(np.zeros((n,15)),np.full((n,15),1.08))
        self.assertTrue(audit_matching(ref,exact,w)['accepted']);self.assertFalse(audit_matching(ref,bad,w)['accepted'])

if __name__=='__main__':unittest.main(verbosity=2)
