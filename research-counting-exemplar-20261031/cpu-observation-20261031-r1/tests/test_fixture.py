import importlib.util
import unittest
import numpy as np


class FixtureTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('fixture'),'fixed texture generator missing')
        import fixture
        return fixture

    def test_repeat_is_exact_and_pixels_and_all_six_rois_are_nonconstant(self):
        m=self.module();a=m.generate();b=m.generate()
        self.assertEqual(a.shape,(800,1333,3));self.assertEqual(a.dtype,np.uint8)
        self.assertTrue(np.array_equal(a,b));self.assertGreater(float(a.std()),15)
        for x1,y1,x2,y2 in m.POSITIVE+m.NEGATIVE:
            self.assertGreater(float(a[y1:y2,x1:x2].std()),10)

    def test_fixture_requires_fixed_seed_and_no_tuning_arguments(self):
        m=self.module()
        with self.assertRaises(TypeError):m.generate(seed=999)
        self.assertEqual(m.TEXTS,{'F':('red toy','blue toy'),'C':('toy',''),'V':('','')})


if __name__=='__main__':unittest.main(verbosity=2)
