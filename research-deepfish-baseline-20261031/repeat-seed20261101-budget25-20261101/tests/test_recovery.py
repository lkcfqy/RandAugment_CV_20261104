import hashlib,json,pathlib,sys,tempfile,unittest,time
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'payload'))
from contracts import validate_context,ResourceUndetermined,ContractError
from transfer import stage_index,validate_ack
class Recovery(unittest.TestCase):
 def context(self):return dict(authorization={'upload':True,'model_run':True},quote_verified=True,persistent_storage_verified=True,external_release_guard_verified=True,cu_per_hour_upper_bound=1.15,allocation_started_epoch=1000,artifact_dir='/content/outputs/run',persistence_mode='local_cli_sha256',persistence_probe={'bytes':8388608,'remote_sha256':'a'*64,'local_sha256':'a'*64,'download_bytes_per_second':2000000},prior_cu=.0569269504676395,prior_gpu_seconds=20.373)
 def test_budget_never_resets_prior_cu_or_gpu_time(self):
  c=self.context();b=validate_context(c,1100);self.assertAlmostEqual(b['hard_deadline_epoch'],8179.627)
 def test_ephemeral_directory_needs_actual_download_hash_receipt(self):
  c=self.context();c['persistence_probe']['local_sha256']='b'*64
  with self.assertRaises(ResourceUndetermined):validate_context(c,1100)
 def test_chunk_index_reassembles_exact_original_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);out=root/'out';out.mkdir();data=bytes(range(251))*3;(out/'checkpoint.pth').write_bytes(data)
   rows=stage_index(out,root/'parts',chunk_bytes=64);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['sha256'],hashlib.sha256(data).hexdigest())
   rebuilt=b''.join((root/'parts'/part['file']).read_bytes() for part in rows[0]['parts']);self.assertEqual(rebuilt,data)
   for part in rows[0]['parts']:self.assertEqual(hashlib.sha256((root/'parts'/part['file']).read_bytes()).hexdigest(),part['sha256'])
 def test_stale_or_unverified_ack_cannot_release_stage(self):
  with self.assertRaises(ContractError):validate_ack({'index_sha256':'old','all_files_atomically_persisted_and_hashed':True},'new')
  with self.assertRaises(ContractError):validate_ack({'index_sha256':'new','all_files_atomically_persisted_and_hashed':False},'new')
  validate_ack({'index_sha256':'new','all_files_atomically_persisted_and_hashed':True},'new')
 def test_failure_finalization_uses_reserved_cleanup_window(self):
  import driver
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);ctx=self.context();ctx['artifact_dir']=str(root/'artifacts');ctx['allocation_started_epoch']=time.time()-10
   ctx['persistence_probe']['download_bytes_per_second']=2000000;hard=time.time()+600;seen=[]
   def persist(out,phase,base,deadline):
    seen.append((phase,deadline-time.monotonic()))
   with patch.object(driver,'verify_payload',return_value={}),patch.object(driver.subprocess,'Popen'),patch.object(driver,'prepare_environment',side_effect=ResourceUndetermined('synthetic setup failure')),patch.object(driver,'persist_stage',side_effect=persist),patch.object(driver,'emit_evidence'):
    code=driver._run_owned(ctx,{'hard_deadline_epoch':hard})
   self.assertEqual(code,4);self.assertEqual([x[0] for x in seen],['final']);self.assertGreater(seen[0][1],480)
 def test_official_pretrained_schema_accepts_only_documented_initializers(self):
  import contracts
  self.assertTrue(hasattr(contracts,'validate_pretrained_key_schema'))
  missing={'dn_query_generator.label_embedding.weight'}|{'bbox_head.cls_branches.'+str(i)+'.log_scale' for i in range(7)}
  extra={'language_model.language_backbone.body.model.embeddings.position_ids','language_model.language_backbone.body.model.pooler.dense.weight','language_model.language_backbone.body.model.pooler.dense.bias'}
  schema=contracts.validate_pretrained_key_schema(missing,extra);self.assertEqual(len(schema['log_scale_keys']),7)
  with self.assertRaises(ContractError):contracts.validate_pretrained_key_schema(missing|{'backbone.patch_embed.projection.weight'},extra)
  with self.assertRaises(ContractError):contracts.validate_pretrained_key_schema(missing,extra|{'unexpected.tensor'})
if __name__=='__main__':unittest.main()
