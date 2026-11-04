import tempfile,unittest
from contracts import *

class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.root=Path(self.tmp.name);self.dims={'dinov2_small':2,'deit_tiny':3}
    def tearDown(self):self.tmp.cleanup()
    def make(self,name):
        d=self.root/name;d.mkdir();D=self.dims[name]
        durable_array(d/'W.npy',np.zeros((360,D,100),np.float64));durable_array(d/'b.npy',np.zeros((360,100),np.float64))
        save(d/'HEAD_IDENTITIES.json',canonical_heads())
        choices=[]
        for e in range(12):
            for B in [8,16]:
                full=[h['index'] for p in POLICIES for h in canonical_heads() if (h['episode'],h['B'],h['policy'],h['fold'])==(e,B,p,-1)]
                choices.append({'episode':e,'B':B,'cv_correct':[1,1,0],'selected_index':0,'full_head_indices':full})
        save(d/'FIT_SEAL.json',{'status':'FIT_SEALED','protocol_sha256':'p','identity_sha256':'i','head_count':360,'choices':choices,'files':{f:sha(d/f) for f in ['W.npy','b.npy','HEAD_IDENTITIES.json']}})
    def test_gate_rejects_one_encoder(self):
        self.make('dinov2_small')
        with self.assertRaises(FileNotFoundError):validate_joint_gate(self.root,'p','i',self.dims)
    def test_gate_accepts_complete_both(self):
        for n in self.dims:self.make(n)
        self.assertEqual(len(validate_joint_gate(self.root,'p','i',self.dims)),2)
    def test_gate_rejects_changed_bytes_and_identity(self):
        for n in self.dims:self.make(n)
        with self.assertRaises(AssertionError):validate_joint_gate(self.root,'p','wrong',self.dims)
        (self.root/'deit_tiny/b.npy').write_bytes(b'broken')
        with self.assertRaises(AssertionError):validate_joint_gate(self.root,'p','i',self.dims)
    def test_compact_100_class_persistence_and_scores(self):
        rng=np.random.default_rng(41);x=rng.normal(size=(513,7)).astype(np.float32);W=rng.normal(size=(7,100));b=rng.normal(size=100)
        full=x.astype(np.float64)@W+b
        pred=np.concatenate([(x[i:i+256].astype(np.float64)@W+b).argmax(1).astype(np.uint8) for i in range(0,len(x),256)])
        self.assertTrue(np.array_equal(pred,full.argmax(1)))
        save_npz(self.root/'fixture.npz',x=x,W=W,b=b,pred=pred)
        with np.load(self.root/'fixture.npz',allow_pickle=False) as z:
            for k,v in [('x',x),('W',W),('b',b),('pred',pred)]:self.assertTrue(np.array_equal(z[k],v))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Integration))
    save(ROOT/'TEST_RESULTS.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'real_head_fits':0,'query_utility':0})
    raise SystemExit(not result.wasSuccessful())
