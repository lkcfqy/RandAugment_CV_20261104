"""Logic verification with fabricated metadata; not research evidence."""
import unittest
from audit_prepare import analyze, retention_bounds


class AuditLogicTests(unittest.TestCase):
    def test_bounds_match_exhaustive_missing_validity_assignments(self):
        # Enumerate possible unknown labels independently of the closed form.
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

    def test_missing_second_gate_is_not_imputed_as_failed(self):
        row = dict(candidate_id='fabricated1', actual_group='classA_contextB',
                   validity='valid', semantic_pass='0', novelty_pass='', retained='0')
        result = analyze([row], 1)['groups']['classA_contextB']
        self.assertIsNone(result['audited_valid_descriptive_rates']['novelty_all'])
        self.assertEqual(result['stage2_missing_on_audited_valid'], 1)

    def test_reject_missing_pool_rows_and_unknown_actual_groups(self):
        row = dict(candidate_id='fabricated1', actual_group='A_B', validity='unknown',
                   semantic_pass='1', novelty_pass='1', retained='1')
        with self.assertRaises(ValueError):
            analyze([row], 2)
        with self.assertRaises(ValueError):
            analyze([dict(row, actual_group='unknown')], 1)

    def test_fixed_gates_commute_but_unconditional_rates_do_not_multiply(self):
        # Two known-valid rows: disjoint gates, each passes half, intersection zero.
        rows = [dict(candidate_id='fabricated'+str(i), actual_group='A_B',
                     validity='valid', semantic_pass=str(c), novelty_pass=str(n),
                     retained=str(c*n)) for i, (c, n) in enumerate([(1, 0), (0, 1)])]
        rates = analyze(rows, 2)['groups']['A_B']['audited_valid_descriptive_rates']
        self.assertEqual(rates['joint'], 0)
        self.assertNotEqual(rates['semantic']*rates['novelty_all'], rates['joint'])
        swapped = [dict(r, semantic_pass=r['novelty_pass'], novelty_pass=r['semantic_pass'])
                   for r in rows]
        self.assertEqual(analyze(swapped, 2)['groups']['A_B']['valid_recall_bounds'],
                         analyze(rows, 2)['groups']['A_B']['valid_recall_bounds'])


if __name__ == '__main__':
    unittest.main()
