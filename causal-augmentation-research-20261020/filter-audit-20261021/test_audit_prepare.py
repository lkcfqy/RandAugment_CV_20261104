"""Logic verification with fabricated metadata; not research evidence."""
import unittest
from audit_prepare import analyze, retention_bounds


def fabricated(uid='fabricated1', **changes):
    row = dict(candidate_id=uid, source_id='fabricated-source', actual_group='A_B',
               label_validity='valid', background_change_success='1', is_copy='0',
               semantic_pass='1', novelty_pass='1', retained='1',
               audit_probability='1', reviewer_disagreement='none')
    row.update(changes)
    return row


def run(rows, count=None):
    return analyze(rows, len(rows) if count is None else count,
                   actual_group_census=True)


class AuditLogicTests(unittest.TestCase):
    def test_bounds_match_exhaustive_missing_target_assignments(self):
        for a in range(4):
            for c in range(4):
                for u in range(4):
                    for v in range(4):
                        rates = [(a+x)/(a+c+x+y)
                                 for x in range(u+1) for y in range(v+1)
                                 if a+c+x+y > 0]
                        b = retention_bounds(a, c, u, v)
                        if rates:
                            self.assertAlmostEqual(b['lower'], min(rates))
                            self.assertAlmostEqual(b['upper'], max(rates))
                        else:
                            self.assertIsNone(b['lower'])
                            self.assertIsNone(b['upper'])

    def test_label_correct_no_change_or_copy_does_not_enter_target_denominator(self):
        rows = [fabricated(),
                fabricated('no-change', background_change_success='0'),
                fabricated('copy', is_copy='1'),
                fabricated('label-invalid', label_validity='invalid')]
        group = run(rows)['groups']['A_B']
        self.assertEqual(group['known_target_eligible'], 1)
        self.assertEqual(group['known_ineligible'], 3)
        self.assertEqual(group['eligible_kept_a'], 1)
        self.assertEqual(group['gate_combinations']['C1_N1']['label_validity']['valid'], 3)

    def test_partial_target_audits_remain_unknown_unless_known_failure(self):
        rows = [fabricated('partial', is_copy=''),
                fabricated('failed', label_validity='unknown', is_copy='',
                           background_change_success='0'),
                fabricated('unreviewed', label_validity='unknown', is_copy='',
                           background_change_success='', reviewer_disagreement='not_reviewed',
                           audit_probability='0.1')]
        group = run(rows)['groups']['A_B']
        self.assertEqual(group['known_target_eligible'], 0)
        self.assertEqual(group['known_ineligible'], 1)
        self.assertEqual(group['target_eligibility_unknown'], 2)
        self.assertEqual(group['target_recall_bounds']['lower'], 1)
        self.assertFalse(group['target_recall_bounds']['target_exists_guaranteed'])

    def test_missing_second_gate_is_not_imputed_as_failed(self):
        row = fabricated(semantic_pass='0', novelty_pass='', retained='0')
        group = run([row])['groups']['A_B']
        self.assertIsNone(group['known_eligible_unweighted_descriptive_rates']['novelty_all'])
        self.assertEqual(group['stage2_missing_on_known_eligible'], 1)
        self.assertEqual(group['gate_combinations']['C0_Nmissing']['candidate_count'], 1)
        self.assertEqual(group['gate_combinations']['C0_N0']['candidate_count'], 0)

    def test_full_table_distinguishes_invalid_samples_rejected_by_different_gates(self):
        a = fabricated(label_validity='invalid', semantic_pass='0', retained='0')
        b = fabricated(label_validity='invalid', novelty_pass='0', retained='0')
        ga = run([a])['groups']['A_B']
        gb = run([b])['groups']['A_B']
        self.assertEqual(ga['known_ineligible'], gb['known_ineligible'])
        self.assertNotEqual(ga['gate_combinations'], gb['gate_combinations'])
        self.assertEqual(ga['gate_combinations']['C0_N1']['label_validity']['invalid'], 1)

    def test_reject_missing_pool_group_census_and_required_design_fields(self):
        row = fabricated()
        with self.assertRaises(ValueError):
            run([row], 2)
        with self.assertRaises(ValueError):
            analyze([row], 1)
        for change in ({'actual_group': 'unknown'}, {'source_id': ''},
                       {'audit_probability': 'nan'}, {'audit_probability': '0'},
                       {'reviewer_disagreement': 'not_reviewed'},
                       {'reviewer_disagreement': 'unresolved'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run([dict(row, **change)])
        with self.assertRaises(ValueError):
            run([row, row])

    def test_fixed_gates_commute_but_unconditional_rates_do_not_multiply(self):
        rows = [fabricated('fabricated'+str(i), semantic_pass=str(c),
                           novelty_pass=str(n), retained=str(c*n))
                for i, (c, n) in enumerate([(1, 0), (0, 1)])]
        rates = run(rows)['groups']['A_B']['known_eligible_unweighted_descriptive_rates']
        self.assertEqual(rates['joint'], 0)
        self.assertNotEqual(rates['semantic']*rates['novelty_all'], rates['joint'])
        swapped = [dict(r, semantic_pass=r['novelty_pass'], novelty_pass=r['semantic_pass'])
                   for r in rows]
        self.assertEqual(run(swapped)['groups']['A_B']['target_recall_bounds'],
                         run(rows)['groups']['A_B']['target_recall_bounds'])
        self.assertEqual(run(rows)['groups']['A_B']['source_count'], 1)


if __name__ == '__main__':
    unittest.main()
