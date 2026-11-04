from common import *
import unittest

class Contracts(unittest.TestCase):
    def test_ln_analytic_finite_difference(self):
        with seeded(1,1):
            z=torch.randn(7,13,dtype=torch.float64);p=torch.randn_like(z);g=torch.randn(13,dtype=torch.float64);b=torch.randn_like(g)
        for eps in (1e-6,.1):
            a=ln_derivative(z,p,g,eps);h=1e-5
            f=(F.layer_norm(z+h*p,(13,),g,b,eps)-F.layer_norm(z-h*p,(13,),g,b,eps))/(2*h)
            self.assertLess(float((a-f).norm()/a.norm()),1e-7)
    def test_constant_position_shift_is_LN_null(self):
        with seeded(2,2):z=torch.randn(5,11,dtype=torch.float64)
        a=ln_derivative(z,torch.ones_like(z)*3,torch.ones(11),1e-6)
        self.assertTrue(torch.equal(a,torch.zeros_like(a)))
    def test_matched_noise_exact_multiset_and_finite(self):
        with seeded(3,3):x=torch.randn(3,100,100)
        y=correlated(x)
        for c in range(3):self.assertTrue(torch.equal(x[c].flatten().sort().values,y[c].flatten().sort().values))
        self.assertFalse(torch.equal(x,y));self.assertTrue(torch.isfinite(y).all())
        self.assertAlmostEqual(var(x),var(y),places=12)
    def test_official_erasing_geometry_and_rng(self):
        row=read('PROTOCOL.json')['images'][0];x=torch.zeros(3,224,224)
        before=torch.get_rng_state().clone();py=random.getstate();v=variants(x,row)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(py,random.getstate())
        t,l,h,w=row['rectangle_tlhw']
        for mode in ('pixel','rand','matched_correlated'):
            mask=torch.zeros_like(x,dtype=torch.bool);mask[:,t:t+h,l:l+w]=True
            self.assertTrue(torch.equal(v[mode][~mask],x[~mask]))
        for c in v['rand'][:,t:t+h,l:l+w]:self.assertEqual(float(c.double().var(unbiased=False)),0.)
    def test_flatten_vs_LN_axis_distinction(self):
        z=torch.tensor([[0.,0.],[2.,2.]])
        self.assertEqual(var(z),1.);self.assertEqual(float(z.var(-1,unbiased=False).mean()),0.)
    def test_known_linear_covariance_identity_not_novel(self):
        with seeded(4,4):x=torch.randn(100,8,dtype=torch.float64);w=torch.randn(3,8,dtype=torch.float64)
        xc=x-x.mean(0);cov=xc.T@xc/100;y=x@w.T
        self.assertTrue(torch.allclose(y.var(0,unbiased=False),(w@cov@w.T).diag(),atol=1e-12,rtol=1e-12))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    dump('TEST_RESULTS.json',{'UTC':utc(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'model_constructions':0,'real_image_forwards':0})
    raise SystemExit(0 if result.wasSuccessful() else 1)
