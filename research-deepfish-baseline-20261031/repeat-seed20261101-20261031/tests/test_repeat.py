import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'payload'))
import contracts as c
class Repeat(unittest.TestCase):
 def timings(self):return dict(init_seconds=30,p90_update_a=2,p90_update_b=3,p90_predict_seconds=.3,checkpoint_bytes=700000000,write_bps=100000000)
 def test_frozen_U_cannot_silently_drop_to50_for_short_budget(self):
  self.assertTrue(hasattr(c,'fixed_repeat_updates'),'fixed-U gate absent')
  r=c.fixed_repeat_updates(self.timings(),700);self.assertEqual(r['status'],'RESOURCE_UNDETERMINED');self.assertEqual(r['updates'],0)
 def test_full_U_fit_remains200_with_no_metric_inputs(self):
  self.assertTrue(hasattr(c,'fixed_repeat_updates'),'fixed-U gate absent')
  self.assertEqual(c.fixed_repeat_updates(self.timings(),2000)['updates'],200)
  t=self.timings();t['dev_AP']=.9
  with self.assertRaises(c.ContractError):c.fixed_repeat_updates(t,2000)
 def test_seed_mismatch_rejected_before_model_imports(self):
  self.assertTrue(hasattr(c,'repeat_training_seed'),'seed contract absent')
  self.assertEqual(c.repeat_training_seed({'randomness':{'seed':20261101}}),20261101)
  for s in [20261031,20261102,True]:
   with self.assertRaises(c.ContractError):c.repeat_training_seed({'randomness':{'seed':s}})
 def test_complete_cost_carries_previous_spend_and_rejects_fast_rate(self):
  self.assertTrue(hasattr(c,'require_complete_repeat_budget'),'full-cost gate absent')
  r=c.require_complete_repeat_budget(1.1001052667892774,3225.2735856513978,1.15,20)
  self.assertGreater(r['remaining_seconds'],2797);self.assertLess(r['remaining_seconds'],2798)
  with self.assertRaises(c.ResourceUndetermined):c.require_complete_repeat_budget(1.1001052667892774,3225.2735856513978,1.5,20)
  with self.assertRaises(c.ResourceUndetermined):c.require_complete_repeat_budget(1.1001052667892774,7100,1.15,20)
if __name__=='__main__':unittest.main()
