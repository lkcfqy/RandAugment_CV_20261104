import unittest
from analyze import *

class AnalysisTests(unittest.TestCase):
    def test_reference_excludes_self_and_stable(self):
        ref=np.eye(6);v=ref[0].copy();self.assertEqual(top5(v,ref,0),[1,2,3,4,5])
    def test_tie_has_no_order_mismatch(self):
        self.assertFalse(ordinary_mismatch([1.,1.001],[2.,1.]));self.assertTrue(ordinary_mismatch([2.,1.],[1.,2.]))
    def test_jaccard_exact(self):
        self.assertEqual(jaccard([1,2],[1,2]),0);self.assertAlmostEqual(jaccard([1,2],[2,3]),2/3)
    def test_slope_linear(self):
        a=np.array([1.,2.,3.]);b=np.array([.1,.2,.3]);f=np.stack([a-.01*b,a,a+.01*b]);self.assertAlmostEqual(output_slope(f),.1,12)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests))
    dump('ANALYSIS_TEST_RESULTS.json',{'UTC':utc(),'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),'model_constructions':0})
    sys.exit(0 if result.wasSuccessful() else 1)
