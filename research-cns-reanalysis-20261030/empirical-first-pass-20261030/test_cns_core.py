"""SYNTHETIC NONEMPIRICAL fixtures: no benchmark measurements or trained models."""
import csv
import io
import random
import unittest
from dataclasses import replace

from cns_core import (SCALES, SHIFTS, MODELS, Row, SchemaError, trajectory_features,
                      assemble, align_models, paired_sources, official_histogram, read_csv_rows)


def make_rows(sequence, shift=SHIFTS[0], class_id=2, seed=1):
    return [Row(shift, class_id, seed, scale, class_id if value else (class_id+1)%1000,
                value, f"class_{class_id:03d}_synthetic", "synthetic",
                f"seed_{seed}_scale_{int(scale*1000)}.jpeg")
            for scale, value in zip(SCALES, sequence)]


class TrajectoryTests(unittest.TestCase):
    def test_all_correct_stays_in_denominator(self):
        tr, _ = assemble(make_rows([True]*6), MODELS[0])
        features = next(iter(tr.values())).features
        self.assertIsNone(features['first_error'])
        self.assertIsNone(features['terminal_error'])
        hist = official_histogram(tr.values())
        self.assertEqual(hist['denominator_all_trajectories'], 1)
        self.assertEqual(sum(hist['counts'].values()), 0)

    def test_all_wrong_includes_scale_zero(self):
        f = trajectory_features([False]*6)
        self.assertEqual((f['first_error'], f['terminal_error']), (0.0, 0.0))
        self.assertFalse(f['recovered_after_first_error'])

    def test_recovery_and_correct_endpoint(self):
        f = trajectory_features([True, False, False, True, True, True])
        self.assertEqual(f['first_error'], 0.5)
        self.assertIsNone(f['terminal_error'])
        self.assertTrue(f['recovered_after_first_error'])
        self.assertEqual((f['correct_to_wrong'], f['wrong_to_correct']), (1, 1))

    def test_terminal_at_target_need_not_first_at_target(self):
        f = trajectory_features([True, False, True, False, False, False])
        self.assertEqual((f['first_error'], f['terminal_error']), (0.5, 1.5))

    def test_multiple_flips(self):
        f = trajectory_features([True, False, True, False, True, False])
        self.assertEqual((f['correct_to_wrong'], f['wrong_to_correct']), (3, 2))
        self.assertEqual(f['terminal_error'], 2.5)

    def test_final_first_error_has_no_recovery_opportunity(self):
        f = trajectory_features([True]*5+[False])
        self.assertEqual(f['first_error'], 2.5)
        self.assertFalse(f['recovery_eligible'])
        self.assertEqual(f['recovery_opportunities'], 0)

    def test_missing_scale_excluded_not_censored(self):
        tr, exclusions = assemble(make_rows([True]*6)[:-1], MODELS[0])
        self.assertEqual(tr, {})
        self.assertEqual(exclusions[0]['reason'], 'incomplete_trajectory_not_censoring')

    def test_duplicate_key_or_variant_rejected(self):
        rows = make_rows([True]*6)
        with self.assertRaises(SchemaError):
            assemble(rows+[rows[0]], MODELS[0])

    def test_unsorted_scale_is_explicitly_sorted(self):
        rows = make_rows([True, True, False, True, False, False])
        random.Random(7).shuffle(rows)
        tr, _ = assemble(rows, MODELS[0])
        self.assertEqual(next(iter(tr.values())).features['first_error'], 1.0)

    def test_model_merge_uses_key_not_row_order(self):
        rows = make_rows([True]*6)
        other = [replace(r, correct=False, pred_class=3) for r in reversed(rows)]
        audit = align_models({MODELS[0]: rows, MODELS[1]: other})
        self.assertEqual(audit['missing_key_counts'], dict.fromkeys(MODELS, 0))
        with self.assertRaises(SchemaError):
            align_models({MODELS[0]:rows, MODELS[1]:[replace(other[0], img_name='wrong.jpeg')]+other[1:]})

    def test_outer_merge_reports_missing_model_keys(self):
        rows = make_rows([True]*6)
        audit = align_models({MODELS[0]: rows, MODELS[1]: rows[:-1]})
        self.assertEqual(audit['missing_key_counts'][MODELS[1]], 1)

    def test_same_eligibility_denominator_on_both_sides(self):
        rows = (make_rows([True]*6, SHIFTS[0], seed=1)+make_rows([False]+[True]*5, SHIFTS[1],seed=1)
                +make_rows([True]*6, SHIFTS[0], seed=2)+make_rows([True]*6, SHIFTS[1], seed=2))
        tr, _ = assemble(rows, MODELS[0])
        common, eligible, audit = paired_sources(tr)
        self.assertEqual(common, {(2,1),(2,2)})
        self.assertEqual(eligible, {(2,2)})
        self.assertEqual(audit['scale0_prediction_mismatch_count'],1)

    def test_schema_bool_and_correctness_are_checked(self):
        fields=['pred_class','gt_class','shift','class_str','img_name','class_name','seed','scale',MODELS[0]]
        def data(flag):
            out=io.StringIO(); w=csv.writer(out); w.writerow(fields)
            w.writerow([2,2,SHIFTS[0],'class_002_synthetic','seed_1_scale_0.jpeg','synthetic',1,0,flag])
            out.seek(0); return out
        self.assertTrue(read_csv_rows(data('True'),MODELS[0])[0].correct)
        for flag in ('False','maybe',''):
            with self.assertRaises(SchemaError): read_csv_rows(data(flag),MODELS[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
