import ast,copy,json,pathlib,subprocess,sys,unittest,tempfile,time,os
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];PAYLOAD=ROOT/'payload';sys.path.insert(0,str(PAYLOAD));sys.path.insert(0,str(ROOT/'test_fixtures'))
try:import image_contracts as ic
except ModuleNotFoundError:ic=None
class Images(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(ic,'image contracts not implemented')
 def test_official_pure_color_identity_and_equal_single_rng_draw(self):
  from pure_color import Color
  result=ic.verify_color_contract(Color)
  self.assertTrue(result['identity_exact']);self.assertTrue(result['rng_equal']);self.assertTrue(result['exactly_one_numpy_draw']);self.assertTrue(result['b_matches_cv2_bgr_blend'])
 def test_color_rejects_float_or_rgb_contract_ambiguity(self):
  import numpy as np
  with self.assertRaises(ValueError):ic.require_bgr_uint8(np.zeros((2,2,3),dtype=np.float32),'cv2')
  with self.assertRaises(ValueError):ic.require_bgr_uint8(np.zeros((2,2,3),dtype=np.uint8),'pillow')
class Configuration(unittest.TestCase):
 def configs(self):
  self.assertTrue((PAYLOAD/'config_A.json').exists(),'explicit configurations not yet generated')
  return [json.loads((PAYLOAD/('config_'+a+'.json')).read_text()) for a in 'AB']
 def test_only_color_factor_and_output_path_differ(self):
  a,b=self.configs();a['work_dir']=b['work_dir']='output'
  for cfg in [a,b]:
   color=cfg['train_dataloader']['dataset']['pipeline'][0]['transforms'][2]
   self.assertEqual((color['type'],color['prob'],color['level']),('Color',1.0,0))
   color['min_mag']=color['max_mag']=1.0
  self.assertEqual(a,b)
 def test_eval_resize_precedes_loading_original_boxes(self):
  for cfg in self.configs():
   p=cfg['test_dataloader']['dataset']['pipeline'];self.assertEqual([x['type'] for x in p],['LoadImageFromFile','FixScaleResize','LoadAnnotations','PackDetInputs']);self.assertIsNone(cfg['val_dataloader']);self.assertIsNone(cfg['default_hooks']['checkpoint'])
 def test_no_official_test_or_unsafe_loading_in_configs(self):
  for cfg in self.configs():
   self.assertEqual(cfg['test_dataloader']['dataset']['ann_file'],'annotations/dev64.json');self.assertEqual(cfg['model']['bbox_head']['num_classes'],1);self.assertEqual(cfg['randomness']['seed'],20261031)
 def test_unapproved_entry_has_no_model_or_network_side_effect(self):
  self.assertTrue((PAYLOAD/'driver.py').exists(),'driver not implemented')
  result=subprocess.run([sys.executable,'-B',str(PAYLOAD/'driver.py'),'--run-context',str(ROOT/'test_fixtures/not_approved.json')],capture_output=True,text=True)
  self.assertEqual(result.returncode,3);self.assertIn('NOT_AUTHORIZED',result.stdout)
  self.assertFalse((PAYLOAD/'runtime_env').exists());self.assertFalse((PAYLOAD/'assets').exists())
 def test_import_worker_does_not_import_model_frameworks(self):
  self.assertTrue((PAYLOAD/'model_worker.py').exists(),'worker not implemented')
  code='import sys;sys.path.insert(0,'+repr(str(PAYLOAD))+');import model_worker;assert "torch" not in sys.modules;assert "mmdet" not in sys.modules'
  self.assertEqual(subprocess.run([sys.executable,'-B','-c',code],capture_output=True).returncode,0)
 def test_json_fields_restore_required_official_tuple_contracts(self):
  import model_worker
  self.assertTrue(hasattr(model_worker,'normalize_config'),'JSON tuple restoration is missing')
  for raw in self.configs():
   cfg=model_worker.normalize_config(raw)
   self.assertEqual(cfg['train_dataloader']['dataset']['metainfo']['classes'],('fish',))
   self.assertEqual(cfg['test_dataloader']['dataset']['metainfo']['classes'],('fish',))
   self.assertEqual(set([cfg['train_dataloader']['dataset']['metainfo']['classes']]),{('fish',)})
   def visit(value):
    if isinstance(value,dict):
     for key,child in value.items():
      if key in ('scale','crop_size'):self.assertIsInstance(child,tuple)
      if key=='scales':self.assertTrue(all(isinstance(v,(tuple,int)) for v in child))
      visit(child)
    elif isinstance(value,(list,tuple)):
     for child in value:visit(child)
   visit(cfg['train_dataloader']['dataset']['pipeline']);visit(cfg['test_dataloader']['dataset']['pipeline'])
   self.assertIsInstance(raw['train_dataloader']['dataset']['metainfo']['classes'],list)
 def test_owned_runtime_released_even_when_payload_or_setup_fails(self):
  import driver
  self.assertTrue(hasattr(driver,'_run_owned'),'whole-owned-lifetime finally boundary missing')
  ctx=dict(authorization={'model_run':True,'upload':True},allocation_started_epoch=time.time()-10,cu_per_hour_upper_bound=2,quote_verified=True,persistent_storage_verified=True,external_release_guard_verified=True,artifact_dir='/mnt/persistent/fixed_run')
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'context.json';p.write_text(json.dumps(ctx));calls=[]
   with patch.dict(os.environ,{'COLAB_RELEASE_TAG':'synthetic-test-only'}),patch.object(driver,'_run_owned',side_effect=driver.ContractError('synthetic payload failure')),patch.object(driver,'native_unassign',side_effect=lambda:calls.append('release') or {'release':'UNCERTAIN'}),patch.object(driver,'emit_evidence'):
    with self.assertRaises(driver.ContractError):driver.run(p)
   self.assertEqual(calls,['release'])
class PackageData(unittest.TestCase):
 def test_original_edge_boxes_retained_and_budget_exact(self):
  p=PAYLOAD/'data/annotations/dev64.json';self.assertTrue(p.exists(),'dev record copy not yet written');d=json.loads(p.read_text());self.assertEqual((len(d['images']),len(d['annotations'])),(64,187));a={x['id']:x for x in d['annotations']}
  self.assertEqual(a[4195]['bbox'],[1867.5,117.5,53.0,117.0]);self.assertEqual(a[4195]['area'],6201.0);self.assertEqual(a[4660]['bbox'],[1779.5,85.5,141.0,83.0]);self.assertEqual(a[4660]['area'],11703.0)
 def test_only_selected_public_images_present(self):
  paths=list((PAYLOAD/'data/images').glob('*.jpg'));self.assertEqual(len(paths),69);self.assertFalse(any('test' in str(p.relative_to(PAYLOAD)) for p in paths))
if __name__=='__main__':unittest.main()
