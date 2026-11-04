"""Synthetic/identity tests only; no encoder or real-data classification."""
from common import *
import unittest,datetime

class Tests(unittest.TestCase):
    def fixture(self,n=7,v=4,d=6,k=3):
        rng=np.random.default_rng(122)
        return rng.normal(size=(n,v,d)),np.arange(n)%k
    def test_explicit_augmented_least_squares(self):
        x,y=self.fixture();n,v,d=x.shape;W,b=ridge(x,y,classes=3)
        A=np.c_[x.reshape(-1,d),np.ones(n*v)]/np.sqrt(n*v)
        Y=np.eye(3)[np.repeat(y,v)]/np.sqrt(n*v)
        reg=np.c_[np.sqrt(LAMBDA)*np.eye(d),np.zeros(d)]
        ref=np.linalg.lstsq(np.r_[A,reg],np.r_[Y,np.zeros((d,3))],rcond=None)[0]
        np.testing.assert_allclose(W,ref[:-1],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(b,ref[-1],rtol=1e-10,atol=1e-10)
    def test_four_identical_views_equal_one(self):
        x,y=self.fixture(v=1);a=ridge(x,y,classes=3);b=ridge(np.repeat(x,4,axis=1),y,classes=3)
        for x,z in zip(a,b): np.testing.assert_allclose(x,z,atol=1e-11,rtol=1e-11)
    def test_per_original_weight_not_number_of_valid_views(self):
        x,y=self.fixture(v=1)
        dup=np.repeat(x,4,axis=1)
        w=np.array([[1,0,0,0],[.5,.5,0,0],[.25]*4,[.2,.3,.1,.4],[0,1,0,0],[0,0,1,0],[0,0,0,1.]])
        a=ridge(x,y,classes=3);b=ridge(dup,y,classes=3,view_weights=w)
        for p,q in zip(a,b): np.testing.assert_allclose(p,q,atol=1e-11,rtol=1e-11)
    def test_normal_equation_gradient_zero(self):
        x,y=self.fixture();W,b=ridge(x,y,classes=3)
        X=x.reshape(-1,x.shape[-1]);Y=np.eye(3)[np.repeat(y,x.shape[1])]
        residual=X@W+b-Y
        np.testing.assert_allclose(X.T@residual/len(X)+LAMBDA*W,0,atol=1e-11)
        np.testing.assert_allclose(residual.mean(0),0,atol=1e-11)
    def test_train_centering_and_repeat_determinism(self):
        x,y=self.fixture();W,b=ridge(x,y,classes=3);W2,b2=ridge(x,y,classes=3)
        self.assertTrue(np.array_equal(W,W2));self.assertTrue(np.array_equal(b,b2))
        np.testing.assert_allclose(x.mean((0,1))@W+b,np.eye(3)[y].mean(0),atol=1e-12)
        # Query data are not an input to the fitting function.
    def test_tie_rule(self):
        self.assertEqual(choose([3,3,3]),0);self.assertEqual(choose([2,3,3]),1)
    def test_identity_nesting_and_grouped_folds(self):
        m=json.loads((ROOT/"IDENTITIES.json").read_text())
        p=json.loads((ROOT/"PROTOCOL.json").read_text())
        self.assertEqual(sha(ROOT/"IDENTITIES.json"),p["identities_sha256"])
        q={x["rgb_sha256"] for x in m["query"]}
        for ep in m["episodes"]:
            rows=ep["support"];B=ep["B"]
            self.assertEqual(len(rows),10*B)
            self.assertFalse(q&{x["rgb_sha256"] for x in rows})
            for c in range(10):
                for f in range(4): self.assertEqual(sum(x["label"]==c and x["fold"]==f for x in rows),B//4)
            for x in rows: self.assertEqual(len({x["fold"] for _ in range(4)}),1)
        for e in range(12):
            a=m["episodes"][2*e]["support"];b=m["episodes"][2*e+1]["support"]
            self.assertTrue({x["member"] for x in a}<={x["member"] for x in b})
            self.assertEqual({x["member"]:x["fold"] for x in a},{x["member"]:x["fold"] for x in b if x["rank_in_class"]<8})
        self.assertEqual(len(m["episodes"])*3*5,360)
    def test_transform_reproducibility_and_rng_isolation(self):
        img=Image.fromarray(np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3))
        state=torch.random.get_rng_state().clone()
        a=make_views(img,"synthetic");b=make_views(img,"synthetic")
        self.assertEqual(len(a),9);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(a,b)))
        for v in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(rng_seed("geometry","synthetic",v));weak=WEAK(img)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(rng_seed("randaugment","synthetic",v));ra=RA(weak.copy())
            self.assertTrue(torch.equal(a[1+v],TENSOR(weak)))
            self.assertTrue(torch.equal(a[5+v],TENSOR(ra)))

if __name__=="__main__":
    t=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    atomic_json("TEST_RESULTS.json",{"status":"PASS" if result.wasSuccessful() else "FAIL","tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"seconds":time.perf_counter()-t,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"model_instantiations":0,"query_predictions":0})
    raise SystemExit(0 if result.wasSuccessful() else 1)

