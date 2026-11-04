import unittest
import numpy as np
from analyze_saved import tie_enumeration,winners,paired_stage_flags

class AnalysisTests(unittest.TestCase):
    def test_all_tie_assignments_are_retained(self):
        r=tie_enumeration([[5,5,4],[3,4,4]],[[10,20,90],[90,30,40]],100)
        self.assertEqual(r['assignment_count'],4)
        self.assertEqual({tuple(a['choice_indices']) for a in r['all_assignments']},{(0,1),(0,2),(1,1),(1,2)})
        self.assertEqual(r['bounds']['accuracy_percent'],[20,30])
        self.assertEqual(r['bounds']['regret_pp'],[60,70])
        self.assertEqual(r['frequency_bounds']['weak'],[0,2])
    def test_untied_keeps_unique_original_choice(self):
        r=tie_enumeration([[6,5,4]],[[100,90,80]],100)
        self.assertEqual(r['assignment_count'],1)
        self.assertEqual(r['bounds']['minus_weak_pp'],[10,10])
    def test_query_best_can_be_tied_without_loss(self):
        self.assertEqual(winners([10,20,20]),[1,2])
        r=tie_enumeration([[1,2,2]],[[10,20,20]],100)
        self.assertEqual(r['bounds']['regret_pp'],[0,0])
    def test_accounting_identity(self):
        full=np.array([[10,20,30],[30,40,50]])
        selected=full[[0,1],[0,2]]
        for p in range(3):
            self.assertEqual(float((selected-full[:,p]).mean()),float((full.max(1)-full[:,p]).mean()-(full.max(1)-selected).mean()))
    def test_third_candidate_does_not_fake_same_pair_evidence(self):
        f=paired_stage_flags(1,[78175,78300,78850],[811,801,803])
        self.assertTrue(f['selected_wrong_full'])
        self.assertTrue(f['selected_wrong_fold'])
        self.assertFalse(f['fold_inferior_to_any_full_optimum'])

if __name__=='__main__':unittest.main()
