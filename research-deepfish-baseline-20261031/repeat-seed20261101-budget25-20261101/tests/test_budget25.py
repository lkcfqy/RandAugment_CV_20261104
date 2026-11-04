"""Catches ledger resets, late bandwidth rejection and inconsistent cost formulas."""
import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'payload'))
import contracts as c

class Budget25(unittest.TestCase):
 def ctx(self):return dict(authorization={'model_run':True,'upload':True},allocation_started_epoch=1000,cu_per_hour_upper_bound=1.15,quote_verified=True,persistent_storage_verified=True,external_release_guard_verified=True,artifact_dir='/mnt/persistent/fixed_run',prior_cu=1.2577442410187722,prior_gpu_seconds=3718.752113674164)
 def test_prior_seconds_remain_the_earlier_boundary_after_CU_extension(self):
  b=c.validate_context(self.ctx(),1020)
  self.assertAlmostEqual(b['hard_deadline_epoch'],4481.247886325836)
  self.assertAlmostEqual(b['work_remaining_seconds'],3281.247886325836)
 def test_approved_spend_over_two_is_not_rejected_or_reset(self):
  ctx=self.ctx();ctx.update(prior_cu=2.1,prior_gpu_seconds=4000)
  try:b=c.validate_context(ctx,1020)
  except c.ResourceUndetermined as e:self.fail('New approved cumulative cap was not applied: '+str(e))
  self.assertAlmostEqual(b['hard_deadline_epoch'],2252.173913043478)
 def gate(self,bps,rate=1.15,elapsed=20):
  self.assertTrue(hasattr(c,'complete_repeat_budget'),'early marker-based component cost gate missing')
  return c.complete_repeat_budget(1.2577442410187722,3718.752113674164,rate,elapsed,bps)
 def test_saved_good_bandwidth_fits_full_U_with_preparation_and_release(self):
  r=self.gate(2354867.827203123)
  self.assertEqual(r['status'],'FITS')
  self.assertAlmostEqual(r['remaining_seconds'],3461.247886325836)
  self.assertAlmostEqual(r['required_seconds'],3162.505627059545,places=5)
  self.assertEqual(r['formal_U'],200)
 def test_slow_marker_blocks_before_any_preparation(self):
  r=self.gate(1500000)
  self.assertEqual(r['status'],'RESOURCE_UNDETERMINED')
  self.assertGreater(r['required_seconds'],r['remaining_seconds'])
 def test_fast_charge_rate_blocks_even_when_GPU_time_is_left(self):
  self.assertEqual(self.gate(2354867.827203123,rate=2)['status'],'RESOURCE_UNDETERMINED')
 def test_elapsed_preparation_is_counted_once(self):
  a=self.gate(2354867.827203123,elapsed=20);b=self.gate(2354867.827203123,elapsed=200)
  self.assertAlmostEqual(a['required_seconds']-b['required_seconds'],180)
  self.assertAlmostEqual(a['remaining_seconds']-b['remaining_seconds'],180)
 def test_invalid_bandwidth_cannot_enter_cost_gate(self):
  self.assertTrue(hasattr(c,'complete_repeat_budget'),'early marker-based component cost gate missing')
  for bps in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(c.ResourceUndetermined):c.complete_repeat_budget(1.2577442410187722,3718.752113674164,1.15,20,bps)

if __name__=='__main__':unittest.main()
