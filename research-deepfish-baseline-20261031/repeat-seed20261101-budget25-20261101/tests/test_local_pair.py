import importlib.util,json,pathlib,sys,tempfile,unittest,hashlib
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gpu-run'))
try:import verify_local_pair as v
except ModuleNotFoundError:v=None
class LocalPair(unittest.TestCase):
 def fixture(self,root,nonfinite=False):
  traces=[{'image_id':i,'steps':[{'geometry':'same','rng':'same'}]} for _ in range(40) for i in (930,1892,1900,2244,2363)]
  for arm in 'AB':
   state={f'tensor{i}':torch.tensor([float(i)]) for i in range(908)}
   if nonfinite and arm=='B':state['tensor0']=torch.tensor([float('inf')])
   path=root/f'final_{arm}_U200.pth';torch.save(state,path)
   report={'seed':20261101,'actual_updates':200,'initial_state_sha256':'same','trainable_names_sha256':'same','optimizer_initial_state_empty':True,'exposures':{'updates':200,'per_image':40,'counts':{str(i):40 for i in (930,1892,1900,2244,2363)}},'final_checkpoint':{'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
   (root/f'formal_{arm}.json').write_text(json.dumps(report));(root/f'formal_{arm}_trace.json').write_text(json.dumps(traces))
 def test_pair_audit_checks_both_complete_weights_and_exact_traces(self):
  self.assertIsNotNone(v,'local pair verifier absent')
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t);self.fixture(p);r=v.verify_pair(p);self.assertEqual(r['status'],'LOCAL_PAIR_VERIFIED');self.assertEqual(len(r['checkpoints']),2);self.assertTrue(all(x['tensor_keys']==908 for x in r['checkpoints']))
   f=p/'formal_B_trace.json';d=json.loads(f.read_text());d[0]['steps'][0]['rng']='drift';f.write_text(json.dumps(d))
   with self.assertRaisesRegex(v.ContractError,'drift'):v.verify_pair(p)
 def test_nonfinite_checkpoint_blocks_pre_evaluation_ack(self):
  self.assertIsNotNone(v,'local pair verifier absent')
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t);self.fixture(p,True)
   with self.assertRaisesRegex(v.ContractError,'nonfinite'):v.verify_pair(p)
if __name__=='__main__':unittest.main()
