import tempfile,unittest
from pathlib import Path

class DiskTests(unittest.TestCase):
    def test_formal_evidence_write_rejects_disk_overrun_before_creation(self):
        from evidence import Evidence
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'existing').write_bytes(b'x'*80)
            evidence=Evidence(root/'output',owned_root=root,disk_cap=100)
            with self.assertRaises(RuntimeError):evidence.write('results/0000.json',{'boxes':['x'*200]},emit=False)
            self.assertFalse((root/'output/results/0000.json').exists())

if __name__=='__main__':unittest.main()
