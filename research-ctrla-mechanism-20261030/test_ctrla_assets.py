import unittest
from ctrla_adapter import *
from train_common import TrainDataset,RORDataset,fit_statistics

class AssetTests(unittest.TestCase):
    def test_disjoint_stratified_indices(self):
        labels=np.load(ROOT/'data'/'train_labels.npy');allidx=[]
        for role,count in [('fit',46000),('controller',1000),('strength_match',1000),('audit',2000)]:
            idx=np.load(ROOT/'data'/f'{role}_indices.npy');self.assertEqual(len(idx),count)
            np.testing.assert_array_equal(np.bincount(labels[idx]),np.full(10,count//10));allidx.extend(idx)
        self.assertEqual(len(set(allidx)),50000)
    def test_no_test_extraction(self):self.assertFalse((ROOT/'data'/'cifar-10-batches-py'/'test_batch').exists())
    def test_challenge_manifest_and_order(self):
        manifest=json.loads((ROOT/'sampling-manifest.json').read_text())
        for role,n in [('strength_match',1000),('audit',2000)]:
            self.assertEqual(sha256(ROOT/manifest[role]['file']),manifest[role]['sha256'])
            data=np.load(ROOT/manifest[role]['file']);order=data['order']
            self.assertEqual(order.shape,(n,15,5));np.testing.assert_array_equal(order[:,:,-1],np.tile(np.arange(15),(n,1)))
            self.assertTrue(all(len(set(row))==5 for row in order.reshape(-1,5)))
            np.testing.assert_array_equal(data['signs'].sum(0),np.zeros((15,5)))
    def test_quantized_round_trip(self):
        image=Image.fromarray(np.tile(np.arange(256,dtype=np.uint8).reshape(16,16,1),(2,2,3)))
        np.testing.assert_array_equal(TF.to_pil_image(TF.to_tensor(image)),image)
    def test_training_exposure_and_ror_layout(self):
        stats=fit_statistics();ds=TrainDataset([0.]*15,[0.]*15,stats['mean'],stats['std'])
        self.assertEqual(len(ds),92000);self.assertEqual(len(ds)%125,0)
        ror=RORDataset();self.assertEqual(len(ror),151000)
        images=np.load(ROOT/'data'/'train_images.npy',mmap_mode='r');idx=np.load(ROOT/'data'/'controller_indices.npy')
        for op in range(15):
            for local in (0,1):
                m=.1*((-1)**local if SIGNED[op] else 1)
                expected=tensor_image(original_op(Image.fromarray(images[idx[local]]),OPS[op],m))
                actual,_=ror[(1+op*10)*1000+local]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
