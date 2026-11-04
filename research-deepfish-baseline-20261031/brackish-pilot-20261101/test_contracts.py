import unittest,copy,math
from contracts import *
class Tests(unittest.TestCase):
    def test_uniform_cap_includes_empty_identity(self):
        self.assertEqual(cap_indices(10),[0,1,3,4,5,6,8,9])
        self.assertEqual(cap_indices(3),[0,1,2])
        for n in range(1,300):
            ids=cap_indices(n);self.assertEqual(len(ids),len(set(ids)));self.assertTrue(all(0<=i<n for i in ids))
    def test_quarters_distinct(self):
        for n in range(2,300):
            a,b=[support_index(n,True,e) for e in (0,1)]
            self.assertLess(a,b);self.assertEqual(a,math.floor(n/4));self.assertEqual(b,math.floor(3*n/4))
        with self.assertRaises(AssertionError):support_index(1,True,0)
    def test_coverage_boundaries(self):
        for v,last in PARTIAL.items():
            self.assertTrue(covered(v,last));self.assertFalse(covered(v,last+1));self.assertFalse(covered(v,0))
    def test_class_and_coordinate_mapping(self):
        original=[{'category_id':c,'bbox':[-2,3,11,7],'area':77,'iscrowd':0} for c in range(1,7)]
        before=copy.deepcopy(original);out=mapped_fish(original,42)
        self.assertEqual(len(out),2);self.assertEqual([a['category_id'] for a in out],[1,1])
        self.assertTrue(all(a['bbox']==[-2,3,11,7] and a['area']==77 for a in out))
        self.assertEqual(original,before);self.assertEqual(mapped_fish([],99),[])
    def test_strict_dev_tie_p0(self):
        self.assertEqual(choose(.3,.3),'P0');self.assertEqual(choose(.3,.30000000000000004),'A');self.assertEqual(choose(.3,.2),'P0')
    def test_decision_failure_modes(self):
        self.assertTrue(decision(.2,.3,.4,.3)['harmful_selection_mismatch'])
        self.assertTrue(decision(.3,.2,.2,.4)['missed_gain'])
        self.assertEqual(decision(.3,.3,.2,.4)['selected_minus_P0'],0)
    def test_invalid_metrics_do_not_select(self):
        with self.assertRaises(AssertionError):choose(-1,.3)
        with self.assertRaises(AssertionError):choose(.3,float('nan'))
    def test_real_frozen_selection(self):
        x=read(P/'SELECTION.json');g=x['groups']
        self.assertEqual([len(g[k]) for k in ['support0','support1','dev','confirm']],[5,5,288,120])
        self.assertFalse({r['filename'] for r in g['support0']}&{r['filename'] for r in g['support1']})
        self.assertEqual(len({r['video'] for r in g['support0']}&{r['video'] for r in g['support1']}),3)
        self.assertTrue(all(covered(r['video'],int(r['frame'])) for rows in g.values() for r in rows))
if __name__=='__main__':unittest.main()
