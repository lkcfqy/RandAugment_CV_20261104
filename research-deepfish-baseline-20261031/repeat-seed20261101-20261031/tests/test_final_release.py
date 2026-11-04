import pathlib,sys,tempfile,json,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gpu-run'))
import external_watchdog as w
class FinalRelease(unittest.TestCase):
 def test_release_marker_requires_verified_final_download_receipt(self):
  self.assertTrue(hasattr(w,'request_final_release'),'prompt final-release trigger absent')
  with tempfile.TemporaryDirectory() as t:
   p=pathlib.Path(t)
   with self.assertRaises((FileNotFoundError,ValueError)):w.request_final_release(p)
   self.assertFalse((p/'REQUEST_RELEASE').exists())
   f=p/'final_download_receipt.json';f.write_text(json.dumps({'phase':'final','all_files_atomically_persisted_and_hashed':False,'index_sha256':'a'*64}))
   with self.assertRaises(ValueError):w.request_final_release(p)
   self.assertFalse((p/'REQUEST_RELEASE').exists())
   f.write_text(json.dumps({'phase':'final','all_files_atomically_persisted_and_hashed':True,'index_sha256':'a'*64}));w.request_final_release(p)
   self.assertEqual(json.loads((p/'REQUEST_RELEASE').read_text())['index_sha256'],'a'*64)
if __name__=='__main__':unittest.main()
