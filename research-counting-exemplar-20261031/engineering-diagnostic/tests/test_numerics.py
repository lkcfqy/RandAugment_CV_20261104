"""Artificial tensors only: no neural modules, images, or model forward."""
import importlib
import unittest
import torch


class NumericalTests(unittest.TestCase):
    def numerics(self):
        self.assertIsNotNone(importlib.util.find_spec('numerics'), 'numerical checks not implemented')
        return importlib.import_module('numerics')

    def sample(self):
        return {'input_ids': torch.tensor([[101,1008,1008,1008,1012,1008,1008,1008,1012,102]]),
                'pred_logits': torch.tensor([[[0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,float('-inf'),float('-inf')]]]),
                'pred_boxes': torch.tensor([[[.5,.5,.2,.2]]]),
                'text_mask': torch.tensor([[True]*10+[False]*2])}

    def test_fixed_native_padding_allowed_but_invalid_active_logit_rejected(self):
        n = self.numerics(); raw = self.sample()
        stats = n.validate(raw)
        self.assertTrue(stats['active_logits_finite'])
        self.assertFalse(stats['full_logits_finite'])
        raw['pred_logits'][0,0,2] = float('-inf')
        with self.assertRaises(n.NumericalFailure): n.validate(raw)

    def test_finite_padding_and_nonfinite_coordinates_are_rejected(self):
        n = self.numerics()
        raw = self.sample(); raw['pred_logits'][0,0,-1] = 0
        with self.assertRaises(n.NumericalFailure): n.validate(raw)
        raw = self.sample(); raw['pred_boxes'][0,0,0] = float('nan')
        with self.assertRaises(n.NumericalFailure): n.validate(raw)

    def test_cpu_relative_tolerance_accepts_small_difference_and_rejects_large_one(self):
        n = self.numerics(); a = self.sample(); b = self.sample()
        b['pred_boxes'][0,0,0] += .0005
        report = n.compare(a, b)
        self.assertTrue(report['tolerance_passed'])
        b['pred_boxes'][0,0,0] += .001
        with self.assertRaises(n.NumericalFailure): n.compare(a, b)

    def test_matching_tolerance_does_not_override_threshold_mask_disagreement(self):
        n = self.numerics(); a = self.sample(); b = self.sample()
        # Both positive and negative scores straddle equality; 0.00005 is within atol.
        a['pred_logits'][0,0,5:10] = 0
        b['pred_logits'][0,0,0] = .00005
        with self.assertRaises(n.NumericalFailure): n.compare(a, b)

    def test_confidence_threshold_and_negative_comparison_are_both_applied(self):
        n = self.numerics(); raw = self.sample()
        self.assertEqual(n.native_mask(raw).tolist(), [True])
        raw['pred_logits'][0,0,:5] = -2
        raw['pred_logits'][0,0,5:10] = -3
        self.assertEqual(n.native_mask(raw).tolist(), [False])
        raw['pred_logits'][0,0,:5] = 0
        raw['pred_logits'][0,0,5:10] = 1
        self.assertEqual(n.native_mask(raw).tolist(), [False])


if __name__ == '__main__': unittest.main(verbosity=2)
