"""Static counterexamples only; no real eval outputs and no model execution."""
import copy,datetime,json,unittest
from pathlib import Path
from decision_guards import tuning_eligibility,assess_control,METHODS

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.params={m:True for m in METHODS}
        self.selection={m:{'sources':{s:{'feasible':True} for s in ('p','q')}} for m in METHODS}
        self.q={'M':1.05,'coverage':.90,'Dhat':.001}
        self.c={'M':1.55,'coverage':.90,'Dhat':.001}
        self.it={}
        for key,size in [('c',[1.5,1.6]),('q',[1.,1.1])]:
            for metric,bounds in [('M',size),('coverage',[.899,.901]),('Dhat',[.0009,.0011])]:
                self.it[key+':'+metric]={'estimable':True,'simultaneous95':bounds}
    def assess(self,tuning=True):return assess_control(self.c,self.q,self.it,'c','q',tuning)
    def test_comparable_positive_fixture(self):
        self.assertTrue(self.assess()['mechanism_beyond_control_established'])
    def test_raps_and_saps_parameter_infeasibility_blocks_B(self):
        for method in ('raps','saps'):
            p=dict(self.params);p[method]=False
            self.assertFalse(tuning_eligibility(p,self.selection)['all_feasible'])
    def test_any_required_temperature_infeasible_blocks_B(self):
        for method in METHODS:
            s=copy.deepcopy(self.selection);s[method]['sources']['p']['feasible']=False
            self.assertFalse(tuning_eligibility(self.params,s)['all_feasible'])
    def test_large_size_separation_cannot_hide_98pct_coverage(self):
        self.c['coverage']=.98;self.c['Dhat']=.008
        self.it['c:coverage']['simultaneous95']=[.979,.981]
        self.it['c:Dhat']['simultaneous95']=[.0079,.0081]
        self.assertFalse(self.assess()['mechanism_beyond_control_established'])
    def test_D_incomparable_alone_blocks_control_exclusion(self):
        self.c['Dhat']=.004;self.it['c:Dhat']['simultaneous95']=[.0039,.0041]
        self.assertFalse(self.assess()['mechanism_beyond_control_established'])
    def test_missing_q_D_propagates_instead_of_typeerror(self):
        self.q['Dhat']=None;r=self.assess()
        self.assertEqual(r['comparison_status'],'UNESTIMABLE_POINT_METRIC')
        self.assertFalse(r['mechanism_beyond_control_established'])
    def test_uncertain_comparability_blocks_even_good_point(self):
        self.it['c:coverage']['simultaneous95']=[.88,.92]
        self.assertFalse(self.assess()['mechanism_beyond_control_established'])
    def test_missing_family_point_does_not_expand_family(self):
        del self.it['c:Dhat'];before=copy.deepcopy(self.it);r=self.assess()
        self.assertEqual(self.it,before);self.assertFalse(r['mechanism_beyond_control_established'])
    def test_control_tuning_infeasible_blocks(self):
        self.assertFalse(self.assess(False)['mechanism_beyond_control_established'])
    def test_unestimable_CI_blocks(self):
        self.it['q:Dhat']={'estimable':False}
        self.assertFalse(self.assess()['mechanism_beyond_control_established'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':result.wasSuccessful(),
      'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
      'actual_eval_metrics_read':False,'GPU_operations':0,'training_updates':0,'inference_views':0,
      'protocol_thresholds_or_CI_family_changed':False}
    (Path(__file__).resolve().parent/'decision-guard-tests.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
