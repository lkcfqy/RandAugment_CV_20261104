import hashlib,pathlib,sys,tempfile,unittest,json,time
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'payload'))
try:import lifecycle as l
except ModuleNotFoundError:l=None
class Lifecycle(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(l,'lifecycle implementation is not present yet')
 def test_atomic_data_is_fsynced_and_readback_hashed(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'report.json';r=l.atomic_bytes(p,b'{"status":"complete"}\n');self.assertEqual(p.read_bytes(),b'{"status":"complete"}\n');self.assertEqual(r['sha256'],hashlib.sha256(p.read_bytes()).hexdigest());self.assertEqual(len(list(pathlib.Path(d).iterdir())),1)
 def test_failed_release_still_reconciles(self):
  calls=[]
  def command(args,timeout):calls.append(args);raise RuntimeError('transport interrupted')
  def snapshot():return {'assignments':[]}
  r=l.release_owned(command,snapshot,'known-endpoint','our-run');self.assertEqual(r['release'],'CONFIRMED');self.assertEqual(len(calls),1)
 def test_ambiguous_release_never_claims_success(self):
  calls=[]
  def command(args,timeout):calls.append(args)
  def snapshot():raise RuntimeError('unavailable')
  self.assertEqual(l.release_owned(command,snapshot,'known-endpoint','our-run')['release'],'UNCERTAIN');self.assertEqual(len(calls),1)
 def test_only_confirmed_own_active_endpoint_retried(self):
  calls=[];states=iter([{'assignments':[{'name':'our-run','endpoint':'known'}]},{'assignments':[]}])
  r=l.release_owned(lambda a,t:calls.append(a),lambda:next(states),'known','our-run');self.assertEqual(r['release'],'CONFIRMED');self.assertEqual(len(calls),2)
 def test_cannot_release_unattributed_other_session(self):
  calls=[];r=l.release_owned(lambda a,t:calls.append(a),lambda:{'assignments':[{'name':'other','endpoint':'other'}]},None,'our-run');self.assertEqual(r['release'],'UNCERTAIN');self.assertEqual(len(calls),1)
 def test_standalone_expiry_rejects_unapproved_context_before_any_side_effect(self):
  from contracts import NotAuthorized
  self.assertTrue(hasattr(l,'expiry_watch'),'expiry context gate missing')
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'context.json';p.write_text(json.dumps({'authorization':{'upload':False,'model_run':False}}))
   with patch.object(l.importlib.util,'find_spec') as lookup,patch.object(l,'atomic_json') as persist,patch.object(l.subprocess,'run') as release,patch.object(l.time,'sleep') as wait:
    with self.assertRaises(NotAuthorized):l.expiry_watch(p)
    lookup.assert_not_called();persist.assert_not_called();release.assert_not_called();wait.assert_not_called()
 def test_expiry_deadline_must_match_verified_allocation_budget(self):
  from contracts import ContractError
  self.assertTrue(hasattr(l,'validate_expiry_context'),'bounded expiry contract missing')
  ctx=dict(authorization={'upload':True,'model_run':True},allocation_started_epoch=1000,cu_per_hour_upper_bound=2,quote_verified=True,persistent_storage_verified=True,external_release_guard_verified=True,artifact_dir='/mnt/persistent/run',hard_deadline_monotonic=4900,work_deadline_monotonic=4720)
  self.assertEqual(l.validate_expiry_context(ctx,1100,500),4900)
  ctx['hard_deadline_monotonic']+=1e6
  with self.assertRaises(ContractError):l.validate_expiry_context(ctx,1100,500)
if __name__=='__main__':unittest.main()
