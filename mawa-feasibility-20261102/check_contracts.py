"""Synthetic arithmetic/shape tests only; never imports or executes upstream code."""
from pathlib import Path
import ast,json,hashlib,time,datetime,unittest,resource
import numpy as np
import torch
P=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)

def mse(x,y):return (x-y).square().mean()
def release_loss(clean,encoded,augmented):return .5*mse(clean,encoded)+1.5*mse(clean,augmented)
def paper_normalized(clean,encoded,views):
    return (mse(clean,encoded)+sum(mse(clean,v) for v in views))/(1+len(views))
def nn_ids(clean,aug):return torch.cdist(aug,clean).argmin(1)
def f(theta,x):
    w1,b1,w2,b2=theta;return torch.relu(x@w1+b1)@w2+b2

class Contracts(unittest.TestCase):
    def setUp(self):torch.manual_seed(11702)
    def test_static_source_branch_and_default(self):
        text=(P/'sources/src__losses__loss_factory.py.txt').read_text();tree=ast.parse(text)
        adds=[ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n,ast.AugAssign) and ast.unparse(n.target)=='total_loss']
        self.assertIn('loss_config.markov_wasserstein * invariance_loss',adds)
        self.assertIn('loss_config.markov_wasserstein * (identity_loss + invariance_loss) / 2.0',adds)
        trainer=(P/'sources/src__train__trainer.py.txt').read_text()
        self.assertIn('torch.cat(collect_aug_encodings, dim=0)',trainer)
        main=ast.parse((P/'sources/main.py.txt').read_text())
        self.assertTrue(any(k.arg=='num_aug_samples' and isinstance(k.value,ast.Constant) and k.value.value==1 for n in ast.walk(main) if isinstance(n,ast.Call) for k in n.keywords))
    def test_s1_release_weights_not_paper_equal_weights(self):
        z=torch.zeros(2,4);e=torch.ones_like(z);a=3*e
        self.assertEqual(float(release_loss(z,e,a)),14.)
        self.assertEqual(float(paper_normalized(z,e,[a])),5.)
        # Reversing residual sizes changes the ratio: not just a loss scale.
        self.assertNotEqual(float(release_loss(z,e,a)/paper_normalized(z,e,[a])),float(release_loss(z,a,e)/paper_normalized(z,a,[e])))
    def test_s3_default_batch_contract_fails(self):
        z=torch.zeros(2,4);a=torch.zeros(6,4)
        with self.assertRaises(RuntimeError):release_loss(z,z,a)
    def test_s3_batch1_broadcast_exception(self):
        z=torch.zeros(1,4);a=torch.ones(3,4)
        self.assertEqual(float(release_loss(z,z,a)),1.5)
    def test_view_major_target_repeat_and_s3_ratio(self):
        z=torch.randn(4,3);e=torch.randn(4,3);views=[torch.randn(4,3) for _ in range(3)]
        aug=torch.cat(views,dim=0);target=torch.cat([z]*3,dim=0)
        fixed=.5*mse(z,e)+1.5*mse(target,aug)
        self.assertTrue(torch.allclose(fixed,2*paper_normalized(z,e,views),atol=1e-14,rtol=1e-14))
        self.assertFalse(torch.equal(target,z.repeat_interleave(3,dim=0)))
    def test_joint_similarity_preserves_euclidean_neighbours(self):
        z=torch.randn(9,4);a=torch.randn(9,4);b=torch.tensor([2.,-1.,.3,.8])
        for scale in (.5,2.):self.assertTrue(torch.equal(nn_ids(z,a),nn_ids(scale*z+b,scale*a+b)))
    def test_wrapper_normalized_loss_and_gradients_equivalent(self):
        theta=[x.requires_grad_() for x in [torch.randn(4,7),torch.randn(7),torch.randn(7,4),torch.randn(4)]]
        z=torch.randn(6,4);a=torch.randn(6,4);loss=release_loss(z,f(theta,z),f(theta,a));grad=torch.autograd.grad(loss,theta)
        for scale,shift in [(1.,torch.tensor([.2,-.3,1.,-.4])),(.5,torch.zeros(4)),(2.,torch.zeros(4))]:
            zp=scale*z+shift;ap=scale*a+shift
            ep=lambda x:scale*f(theta,(x-shift)/scale)+shift
            lp=release_loss(zp,ep(zp),ep(ap))
            gp=torch.autograd.grad(lp/scale**2,theta)
            self.assertTrue(torch.allclose(lp/scale**2,loss,atol=1e-11,rtol=1e-12))
            self.assertTrue(all(torch.allclose(x,y,atol=1e-11,rtol=1e-12) for x,y in zip(grad,gp)))
    def test_native_mlp_conjugate_parameter_map(self):
        w1,b1,w2,b2=[torch.randn(*s) for s in [(4,7),(7,),(7,4),(4,)]];x=torch.randn(6,4);a=2.;b=torch.randn(4)
        mapped=[w1/a,b1-b@w1/a,a*w2,a*b2+b]
        self.assertTrue(torch.allclose(f(mapped,a*x+b),a*f([w1,b1,w2,b2],x)+b,atol=1e-12,rtol=1e-12))
    def test_loss_multiplier_optimizer_arithmetic_not_training(self):
        # One algebraic state transition on arbitrary arrays; no optimizer or training loop.
        theta=torch.randn(5);g=torch.randn(5);v=torch.rand(5);m=torch.randn(5);c=4.;lr=.001;decay=.0001;eps=1e-8
        sgd=theta-lr*g;scaled_sgd=theta-(lr/c)*(c*g)
        self.assertTrue(torch.equal(sgd,scaled_sgd))
        adamw=theta*(1-lr*decay)-lr*m/(v.sqrt()+eps)
        scaled_adamw=theta*(1-lr*decay)-lr*(c*m)/((c*c*v).sqrt()+c*eps)
        self.assertTrue(torch.equal(adamw,scaled_adamw))
    def test_centered_gram_condition_number_unchanged(self):
        z=torch.randn(18,4);b=torch.randn(4);a=3.
        gram=lambda x:(x-x.mean(0)).T@(x-x.mean(0))
        self.assertTrue(torch.allclose(gram(a*z+b),a*a*gram(z),atol=1e-12,rtol=1e-12))
        self.assertAlmostEqual(float(torch.linalg.cond(gram(z))),float(torch.linalg.cond(gram(a*z+b))),11)
    def test_weighted_affine_closed_form_similarity(self):
        z=np.random.default_rng(11702).normal(size=(14,3));aug=z+np.random.default_rng(3).normal(size=z.shape)*.3
        weights=np.r_[np.full(14,.5),np.full(14,1.5)]
        def solve(x,y):
            design=np.c_[x,np.ones(len(x))];r=np.sqrt(weights)[:,None]
            B=np.linalg.lstsq(design*r,y*r,rcond=None)[0]
            return design@B
        x=np.r_[z,aug];y=np.r_[z,z];pred=solve(x,y);a=2.;b=np.array([.3,-.5,.1]);mapped=(solve(a*x+b,a*y+b)-b)/a
        self.assertTrue(np.allclose(pred,mapped,rtol=1e-12,atol=1e-12))
    def test_exact_duplicate_conflicting_target_bound(self):
        a,b=.5,1.5;y1,y2=0.,2.;best=(a*y1+b*y2)/(a+b)
        attained=a*(best-y1)**2+b*(best-y2)**2
        self.assertEqual(attained,a*b/(a+b)*(y1-y2)**2)
        self.assertGreater(attained,0.)
    def test_nearest_class_collision_is_not_irreducible_loss(self):
        # Deliberately tiny fixture, not an empirical result or proposed augmentation.
        clean=torch.tensor([[-1.,0.],[1.,0.]]);aug=torch.tensor([[1.,.5],[-1.,-.5]])
        self.assertTrue(torch.equal(nn_ids(clean,aug),torch.tensor([1,0])))
        E=lambda z:torch.stack([z[:,0]-4*z[:,1],torch.zeros(len(z))],1)
        self.assertEqual(float(release_loss(clean,E(clean),E(aug))),0.)

if __name__=='__main__':
    start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests':r.testsRun,'errors':len(r.errors),'failures':len(r.failures),'passed':r.wasSuccessful(),'seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'numpy':np.__version__,'torch':torch.__version__,'synthetic_only':True,'upstream_code_imports_or_execution':0,'real_feature_generation':0,'adapter_training_runs':0,'optimizer_instances_or_steps':0}
    (P/'TEST_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
