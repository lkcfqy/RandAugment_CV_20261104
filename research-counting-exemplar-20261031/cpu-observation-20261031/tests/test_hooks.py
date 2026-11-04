import importlib.util
import unittest
import torch


class InstrumentationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('hooks'),'diagnostic instrumentation missing')
        import hooks
        return hooks

    def test_cpu_snapshot_survives_original_inplace_change(self):
        m=self.module();x=torch.tensor([1.,2.]);s=m.snapshot(x);old=m.tensor_hash(s);x.add_(10)
        self.assertEqual(s.tolist(),[1.,2.]);self.assertEqual(m.tensor_hash(s),old)

    def test_hash_binds_dtype_shape_and_exact_bytes(self):
        m=self.module();x=torch.tensor([1.,2.])
        self.assertEqual(m.tensor_hash(x),m.tensor_hash(x.clone()))
        self.assertNotEqual(m.tensor_hash(x),m.tensor_hash(x.reshape(1,2)))
        self.assertNotEqual(m.tensor_hash(x),m.tensor_hash(x.double()))
        self.assertNotEqual(m.tensor_hash(x),m.tensor_hash(x+.000001))

    def test_stage_difference_records_original_tolerance_not_new_one(self):
        m=self.module();a=torch.tensor([0.,1.]);b=torch.tensor([.00005,1.0005])
        self.assertTrue(m.compare_stage(a,b)['within_tolerance'])
        b[0]=.001
        self.assertFalse(m.compare_stage(a,b)['within_tolerance'])

    def test_expected_nonfinite_mask_must_match(self):
        m=self.module();a=torch.tensor([1.,float('-inf')]);b=a.clone()
        self.assertTrue(m.compare_stage(a,b)['within_tolerance'])
        b[-1]=float('inf');self.assertFalse(m.compare_stage(a,b)['within_tolerance'])

    def test_topk_order_change_is_reported_even_when_sets_equal(self):
        m=self.module();a=torch.tensor([2.,2.,1.]);r=m.topk_report(a,torch.tensor([0,1]),a,torch.tensor([1,0]))
        self.assertTrue(r['sets_equal']);self.assertFalse(r['order_equal']);self.assertEqual(r['different_positions'],2)
        self.assertEqual(r['cpu']['cutoff_gap_to_best_excluded'],1.0)
        self.assertEqual(r['cpu']['ties_at_cutoff'],2)

    def test_topk_set_change_and_boundary_gap_not_hidden_by_matching(self):
        m=self.module();a=torch.tensor([3.,2.,2.]);b=torch.tensor([3.,2.,2.00001]);r=m.topk_report(a,torch.tensor([0,1]),b,torch.tensor([0,2]))
        self.assertFalse(r['sets_equal']);self.assertEqual(r['symmetric_difference_size'],2)
        self.assertEqual(r['cpu']['cutoff_gap_to_best_excluded'],0.0)
        self.assertGreater(r['max_absolute_score_delta'],0)


if __name__=='__main__':unittest.main(verbosity=2)
