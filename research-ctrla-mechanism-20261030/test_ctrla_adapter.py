import unittest,json,random
import numpy as np
import torch
from PIL import Image
from ctrla_adapter import *

class RenderingFixtures(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(7);self.img=Image.fromarray(r.integers(0,256,(32,32,3),dtype=np.uint8))
    def test_affine_identity(self):
        for mode in 'SEF':
            actual=render(self.img,list(range(6)),np.zeros(6),mode)
            np.testing.assert_array_equal(actual,self.img)
    def test_zero_photometric_preserves_official_semantics(self):
        expected=self.img
        for name in OPS:expected=original_op(expected,name,0.)
        for mode in 'SEF':np.testing.assert_array_equal(render(self.img,list(range(15)),np.zeros(15),mode),expected)
    def test_single_operation_all_modes(self):
        for i in range(15):
            for m in ((-.63,.47) if SIGNED[i] else (.47,)):
                expected=original_op(self.img,OPS[i],m)
                for mode in 'SEF':np.testing.assert_array_equal(render(self.img,[i],[m],mode),expected)
    def test_original_coordinate_matrices(self):
        for name in OPS[:6]:
            for m in (-1.,-.63,-.125,0.,.125,.47,1.):
                expected=np.asarray(original_op(self.img,name,m)).astype(float)
                actual=np.asarray(matrix_render(self.img,inverse_matrix(name,m))).astype(float)
                self.assertLessEqual(np.max(np.abs(expected-actual)),1.,(name,m))
    def test_integer_translation_inverse_boundary(self):
        order=[0,0];mags=[.5,-.5]
        for mode in ('E','F'):np.testing.assert_array_equal(render(self.img,order,mags,mode),self.img)
        s=np.array(render(self.img,order,mags,'S'))
        self.assertTrue(np.all(s[:,-8:]==0));self.assertGreater(np.abs(s.astype(float)-np.array(self.img)).mean(),0)
    def test_order_and_inverse_composition(self):
        m1=inverse_matrix('TranslateX',.5);m2=inverse_matrix('ShearY',.5)
        self.assertFalse(np.allclose(m1@m2,m2@m1))
        p=np.array([12.5,18.5,1.]);np.testing.assert_allclose((m1@m2)@np.linalg.inv(m1@m2)@p,p,atol=1e-10)
        for name in ('ShearX','ShearY','Rotate'):
            np.testing.assert_allclose(inverse_matrix(name,.25)@inverse_matrix(name,-.25),np.eye(3),atol=1e-12)
    def test_photometric_barrier(self):
        for mode in 'SEF':
            expected=original_op(original_op(original_op(self.img,'TranslateX',.5),'Brightness',-.4),'TranslateX',-.5)
            np.testing.assert_array_equal(render(self.img,[0,7,0],[.5,-.4,-.5],mode),expected)
    def test_quantile_and_schedule(self):
        q=np.linspace(.01,.99,1000)
        for a in (0.,.3,1.):
            t=tilted_quantile(q,a);np.testing.assert_allclose((1-a)*t+a*t*t,q,atol=1e-12)
        self.assertEqual(lr_at(1),.05);self.assertGreater(lr_at(10),lr_at(20))
    def test_momentum_reset_and_gain(self):
        model=torch.nn.Linear(2,2);opt=new_optimizer(model,1)
        model(torch.ones(2,2)).sum().backward();opt.step()
        self.assertTrue(opt.state);self.assertFalse(new_optimizer(model,2).state)
        self.assertEqual(update_xi(.9,.5,1),(.9,0.))
        self.assertAlmostEqual(update_xi(.9,3.,2)[0],.99)
    def test_official_op_pool(self):
        self.assertEqual(len(OPS),15);self.assertEqual(sum(SIGNED),11)

if __name__=='__main__':unittest.main(verbosity=2)
