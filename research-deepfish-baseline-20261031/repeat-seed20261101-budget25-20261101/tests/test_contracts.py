import importlib.util,json,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'payload'))
try:import contracts as c
except ModuleNotFoundError:c=None

class Contracts(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(c,'contracts implementation is not present yet')
 def context(self):return dict(authorization={'model_run':True,'upload':True},allocation_started_epoch=1000,cu_per_hour_upper_bound=2.0,quote_verified=True,persistent_storage_verified=True,external_release_guard_verified=True,artifact_dir='/mnt/persistent/fixed_run')
 def timings(self):return dict(init_seconds=30,p90_update_a=2,p90_update_b=3,p90_predict_seconds=.3,checkpoint_bytes=700000000,write_bps=100000000)
 def test_authorization_checked_before_any_optional_runtime_information(self):
  with self.assertRaises(c.NotAuthorized):c.validate_context({},1100)
 def test_missing_rate_means_undetermined_not_guessed(self):
  ctx=self.context();ctx.pop('cu_per_hour_upper_bound')
  with self.assertRaises(c.ResourceUndetermined):c.validate_context(ctx,1100)
 def test_preparation_is_charged_from_allocation_start(self):
  b=c.validate_context(self.context(),1100);self.assertEqual(b['hard_deadline_epoch'],5500);self.assertEqual(b['work_remaining_seconds'],4220)
 def test_slow_cu_rate_still_caps_two_gpu_hours(self):
  ctx=self.context();ctx['cu_per_hour_upper_bound']=.25;self.assertEqual(c.validate_context(ctx,1100)['hard_deadline_epoch'],8200)
 def test_unverified_persistence_or_release_blocks(self):
  for key in ['persistent_storage_verified','external_release_guard_verified','quote_verified']:
   ctx=self.context();ctx[key]=False
   with self.assertRaises(c.ResourceUndetermined):c.validate_context(ctx,1100)
 def test_expired_budget_rejected(self):
  with self.assertRaises(c.ResourceUndetermined):c.validate_context(self.context(),5400)
 def test_u_frozen_from_timing_only_before_metrics(self):
  self.assertEqual(c.choose_updates(self.timings(),2000)['updates'],200)
  self.assertEqual(c.choose_updates(self.timings(),700)['updates'],50)
  t=self.timings();t['loss']=.001
  with self.assertRaises(c.ContractError):c.choose_updates(t,2000)
 def test_no_fit_keeps_sample_protocol_and_reports_zero(self):
  self.assertEqual(c.choose_updates(self.timings(),100)['status'],'RESOURCE_UNDETERMINED')
 def test_exact_equal_exposures_required(self):
  ids=[930,1892,1900,2244,2363]*5;c.verify_exposures(ids,25)
  ids[-1]=930
  with self.assertRaises(c.ContractError):c.verify_exposures(ids,25)
 def test_preflight_fifteen_exposures_accounted_separately(self):
  c.verify_exposures([930,1892,1900,2244,2363]*3,15)
  with self.assertRaises(c.ContractError):c.verify_exposures([930]*15,15)
 def test_geometry_or_rng_drift_is_not_ignored(self):
  a=[dict(image_id=930,steps=[{'gt':[[1,2,3,4]],'rng':'x'}])];b=json.loads(json.dumps(a));c.verify_paired_traces(a,b)
  b[0]['steps'][0]['rng']='y'
  with self.assertRaises(c.ContractError):c.verify_paired_traces(a,b)
 def test_unknown_or_nonfinite_timings_rejected(self):
  for v in [0,-1,float('inf'),float('nan')]:
   t=self.timings();t['write_bps']=v
   with self.assertRaises(c.ContractError):c.choose_updates(t,2000)
 def test_initialization_contract_includes_trainable_parameters_and_optimizer(self):
  self.assertTrue(hasattr(c,'verify_initializations'),'initialization gate missing')
  good={'initial_state_sha256':'a','trainable_names_sha256':'b','optimizer_initial_state_empty':True};c.verify_initializations([good,dict(good)])
  bad=dict(good,trainable_names_sha256='changed')
  with self.assertRaises(c.ContractError):c.verify_initializations([good,bad])

if __name__=='__main__':unittest.main()
