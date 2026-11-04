import importlib.util,pathlib,sys,tempfile,unittest,shutil,json
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
try:import package_local as p
except ModuleNotFoundError:p=None
class Packaging(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(p,'packager not implemented')
 def test_private_paths_and_tokens_rejected(self):
  for text in ['/Users/private-person/file','/home/person/.aws','ghp_'+'A'*30,'sk-'+'A'*30]:
   with self.assertRaises(ValueError):p.check_text(text)
 def test_no_execution_permission_added_to_example(self):
  import json
  e=json.loads((ROOT/'payload/run_context.example.json').read_text());self.assertFalse(e['authorization']['upload']);self.assertFalse(e['authorization']['model_run'])
 def test_payload_has_exact_expected_types_and_69_original_jpegs(self):
  result=p.audit_payload(ROOT/'payload');self.assertEqual(result['images'],69);self.assertEqual(result['private_metadata_images'],[]);self.assertTrue(result['all_image_hashes_match'])
 def test_archive_roundtrip_receipt_stays_within_thirteen_mib(self):
  try:r=p.build()
  except TypeError as error:self.fail('Archive receipt construction failed: '+str(error))
  self.assertLessEqual(r['archive_bytes'],13*1024**2);self.assertEqual(len(r['archive_sha256']),64);self.assertTrue(r['zip_roundtrip_verified']);self.assertFalse(r['uploaded'])
 def test_changed_annotation_or_selected_membership_cannot_be_recertified(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d)/'payload';shutil.copytree(ROOT/'payload',root)
   ann=root/'data/annotations/dev64.json';original=ann.read_bytes();data=json.loads(original)
   data['annotations'][0]['area']+=1;ann.write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,'annotation'):p.audit_payload(root)
   ann.write_bytes(original);selection=root/'data_selection.json';data=json.loads(selection.read_text());data['dev_ids'][0]=-1;selection.write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,'membership'):p.audit_payload(root)
 def test_receipt_describes_manifest_files_without_claiming_missing_checksums(self):
  result=p.audit_payload(ROOT/'payload');self.assertNotIn('files_without_checksum_metadata',result);self.assertEqual(result['manifest_payload_file_count'],89)
if __name__=='__main__':unittest.main()
