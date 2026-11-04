"""Counterexamples use invented counts only; no dataset or model imports."""
from fractions import Fraction
import importlib.util
import unittest


def artificial_pair(pair_id, f=10, c=31, v=31):
    return {'pair_id':pair_id,'images':[{'image_id':pair_id+'/'+suffix,'truth':{'A':50,'B':50},
        'predictions':{'F':{'A':50+f,'B':50},'C':{'A':50+c,'B':50},'V':{'A':50+v,'B':50}}} for suffix in ('1','2')]}


def policy(ids, na_ids=()):
    return [{'pair_id':p,'eligible':True,'image_ids':[p+'/1',p+'/2']} for p in ids]+[{'pair_id':p,'eligible':False,'reason':'artificial NA fixture','image_ids':[p+'/1',p+'/2']} for p in na_ids]


class MetricTests(unittest.TestCase):
    def metrics(self):
        self.assertIsNotNone(importlib.util.find_spec('metrics'),'paired metric implementation missing')
        import metrics
        return metrics

    def test_joint_A_B_denominator_is_not_mean_of_separate_relative_errors(self):
        m=self.metrics()
        self.assertEqual(m.image_error(1,9,2,9),Fraction(1,10))

    def test_opposite_count_errors_do_not_cancel_through_total_count(self):
        m=self.metrics()
        self.assertEqual(m.image_error(1,9,2,8),Fraction(1,5))
        self.assertEqual(m.image_error(10,10,20,0),Fraction(1,1))

    def test_zero_total_invalid_counts_and_nonfinite_predictions_rejected(self):
        m=self.metrics()
        for args in [(0,0,0,0),(-1,2,0,0),(1,2,-1,0),(1,2,float('nan'),0),(True,2,1,2),(1,2,1.5,2)]:
            with self.subTest(args=args),self.assertRaises(ValueError):m.image_error(*args)

    def test_large_error_not_clipped_or_saturation_excluded(self):
        m=self.metrics()
        self.assertEqual(m.image_error(50,50,1000,50),Fraction(19,2))

    def test_pair_average_weights_images_equally_not_by_object_total(self):
        m=self.metrics();p=artificial_pair('p',0,0,0)
        p['images'][0]['truth']={'A':5,'B':5}
        p['images'][0]['predictions']={x:{'A':5,'B':5} for x in ('F','C','V')}
        p['images'][1]['predictions']['F']={'A':150,'B':50}
        r=m.summarize([p],policy(['p']))
        self.assertEqual(r['pairs'][0]['mean_error_exact']['F'],[1,2])

    def test_oracle_min_after_image_mean_not_before(self):
        m=self.metrics();p=artificial_pair('p',10,0,100)
        p['images'][1]['predictions']['C']={'A':150,'B':50}
        p['images'][1]['predictions']['V']={'A':50,'B':50}
        r=m.summarize([p],policy(['p']))
        self.assertEqual(r['G_exact'],[2,5])

    def test_oracle_min_per_pair_not_global_min_of_mode_means(self):
        m=self.metrics();p1=artificial_pair('p1',0,0,100);p2=artificial_pair('p2',0,100,0)
        r=m.summarize([p1,p2],policy(['p1','p2']))
        self.assertEqual(r['G_exact'],[0,1])

    def test_both_gates_needed_even_with_ten_qualifying_pairs(self):
        m=self.metrics();ids=[f'p{i}' for i in range(41)]
        rows=[artificial_pair(p,10,31,31) if i<10 else artificial_pair(p,10,10,10) for i,p in enumerate(ids)]
        r=m.summarize(rows,policy(ids))
        self.assertEqual(r['qualifying_pairs'],10)
        self.assertEqual(r['G_exact'],[21,410])
        self.assertFalse(r['gate_G']);self.assertTrue(r['gate_pair_count']);self.assertFalse(r['numeric_investment_gate'])

    def test_C_V_boundary_strict_and_F_boundary_inclusive(self):
        m=self.metrics();ids=[f'p{i}' for i in range(41)]
        rows=[artificial_pair(p,10,31,31) if i<9 else artificial_pair(p,10,30,31) for i,p in enumerate(ids)]
        r=m.summarize(rows,policy(ids))
        self.assertTrue(r['gate_G']);self.assertEqual(r['qualifying_pairs'],9)
        self.assertFalse(r['numeric_investment_gate'])
        rows=[artificial_pair(p,10,31,31) for p in ids]
        r=m.summarize(rows,policy(ids));self.assertEqual(r['qualifying_pairs'],41);self.assertTrue(r['numeric_investment_gate'])
        rows[0]=artificial_pair(ids[0],11,31,31)
        self.assertEqual(m.summarize(rows,policy(ids))['qualifying_pairs'],40)

    def test_G_boundary_is_inclusive(self):
        m=self.metrics();r=m.summarize([artificial_pair('p',10,20,20)],policy(['p']))
        self.assertEqual(r['G_exact'],[1,10]);self.assertTrue(r['gate_G'])

    def test_NA_is_reported_without_zero_imputation_or_denominator_dilution(self):
        m=self.metrics();ids=[f'p{i}' for i in range(41)];na=[f'na{i}' for i in range(6)]
        r=m.summarize([artificial_pair(p) for p in ids],policy(ids,na))
        self.assertEqual(r['G_exact'],[21,100]);self.assertEqual(r['eligible_pairs'],41);self.assertEqual(len(r['pairs']),47)
        self.assertTrue(all(x['mean_error'] is None for x in r['pairs'] if x['status']=='NA'))

    def test_missing_extra_duplicate_or_one_image_pairs_never_silently_dropped(self):
        m=self.metrics();p=artificial_pair('p')
        bad=[[p,p],[],[artificial_pair('unexpected')],[dict(p,images=p['images'][:1])],[dict(p,images=[p['images'][0],p['images'][0]])]]
        for rows in bad:
            with self.subTest(rows=rows),self.assertRaises(ValueError):m.summarize(rows,policy(['p']))

    def test_missing_target_or_mode_is_not_zero_imputed(self):
        m=self.metrics();p=artificial_pair('p');del p['images'][0]['predictions']['F']['B']
        with self.assertRaises(ValueError):m.summarize([p],policy(['p']))
        p=artificial_pair('p');del p['images'][0]['predictions']['C']
        with self.assertRaises(ValueError):m.summarize([p],policy(['p']))


if __name__=='__main__':unittest.main(verbosity=2)
