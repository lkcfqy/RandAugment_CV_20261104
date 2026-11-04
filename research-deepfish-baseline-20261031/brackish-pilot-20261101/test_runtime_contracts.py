import unittest,copy,contextlib,io
from contracts import *
from run_pilot import estimate
class Tests(unittest.TestCase):
    def test_historical_control_hash_newline(self):
        groups=read(R/'cpu-paired-AM-20261101/formal_A/OPTIMIZER_GROUPS.json')
        expected=read(P/'PROTOCOL.json')['expected_optimizer_groups_sha256']
        self.assertEqual(hashlib.sha256(canonical(groups)+b'\n').hexdigest(),expected)
        self.assertNotEqual(digest(groups),expected)
    def test_resource_gate_joint_conditions(self):
        p=read(P/'PROTOCOL.json')
        sample={'status':'PREFLIGHT_COMPLETE','completed_updates':1,'peak_RSS_bytes':4*2**30,'steps':[{'shape_CHW':[3,750,1333],'train_step_seconds':8,'pipeline_seconds':.01}]}
        good=estimate({'A0':sample,'A1':sample},30,p)
        self.assertTrue(good['time_pass'] and good['memory_pass'])
        bad=copy.deepcopy(sample);bad['steps'][0]['train_step_seconds']=25
        self.assertFalse(estimate({'A0':bad,'A1':sample},30,p)['time_pass'])
        bad=copy.deepcopy(sample);bad['peak_RSS_bytes']=11*2**30
        self.assertFalse(estimate({'A0':bad,'A1':sample},30,p)['memory_pass'])
    def test_native_metric_maxdets_and_empty_frames(self):
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        g=COCO();g.dataset={'info':{},'images':[{'id':1,'width':100,'height':100},{'id':2,'width':100,'height':100}],'categories':[{'id':1,'name':'fish'}],'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[10,10,10,10],'area':100,'iscrowd':0}]}
        with contextlib.redirect_stdout(io.StringIO()):
            g.createIndex()
            rows=[{'image_id':1,'category_id':1,'bbox':[70,70,5,5],'score':1-i*.001} for i in range(100)]
            rows.append({'image_id':1,'category_id':1,'bbox':[10,10,10,10],'score':.5})
            rows.append({'image_id':2,'category_id':1,'bbox':[10,10,10,10],'score':.9})
            d=g.loadRes(rows);e=COCOeval(g,d,'bbox');e.params.maxDets=[100,300,1000];e.params.imgIds=[1,2];e.params.catIds=[1];e.evaluate();e.accumulate();e.summarize()
        self.assertEqual(e.stats[0],0.)
        self.assertGreater(e.stats[1],0.)
        self.assertEqual(e.params.imgIds,[1,2]);self.assertEqual(len(e.stats),12)
    def test_color_identity_and_coordinate_preservation(self):
        import numpy as np,cv2,mmcv
        from mmdet.datasets.transforms import Color
        mmcv.use_backend('cv2');image=np.random.RandomState(17).randint(0,256,(12,20,3),dtype=np.uint8)
        result={'img':image.copy(),'img_shape':(12,20),'instances':[{'bbox':[1,2,4,8],'bbox_label':0}]};original=copy.deepcopy(result)
        out=Color(prob=1.,level=0,min_mag=1.,max_mag=1.)(result)
        self.assertTrue(np.array_equal(out['img'],original['img']));self.assertEqual(out['instances'],original['instances']);self.assertEqual(out['img_shape'],original['img_shape'])
if __name__=='__main__':unittest.main()
