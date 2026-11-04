"""SYNTHETIC NONEMPIRICAL asset/schema/identity fixtures."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
import zipfile
from cns_core import MODELS, SHIFTS, SchemaError
from cns_assets import (audit_annotations, load_prediction_zip, check_predictions_against_annotations,
                        verify_local_baseline_manifest)
from run_reanalysis import make_synthetic_assets


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.archive,self.metadata=make_synthetic_assets(self.root)
        self.audit,self.keys,_=audit_annotations(self.metadata)

    def tearDown(self): self.temp.cleanup()

    def test_release_shape_zip_and_metadata_agree(self):
        rows,audit=load_prediction_zip(self.archive)
        matching=check_predictions_against_annotations(rows,self.keys)
        self.assertEqual(len(audit['selected_files']),4)
        self.assertEqual(matching[MODELS[0]]['annotation_keys_without_prediction'],1)
        self.assertEqual(matching[MODELS[1]]['annotation_keys_without_prediction'],0)

    def test_missing_fixed_model_does_not_select_another(self):
        empty=self.root/'empty.zip'
        with zipfile.ZipFile(empty,'w'): pass
        with self.assertRaises(SchemaError): load_prediction_zip(empty)

    def test_duplicate_annotation_key_rejected(self):
        data=json.loads(self.metadata.read_text()); data['images'].append(data['images'][0])
        self.metadata.write_text(json.dumps(data))
        with self.assertRaises(SchemaError): audit_annotations(self.metadata)

    def manifest(self,seeds=(1,),different=False,escape=False):
        path=self.root/'baseline_manifest.csv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['shift','gt_class','seed','local_path']); writer.writeheader()
            for seed in seeds:
                for shift in SHIFTS:
                    name=f'seed_{seed:04d}_scale_0000.jpg'
                    image=self.root/shift/'002_synthetic'/name
                    image.parent.mkdir(parents=True,exist_ok=True)
                    image.write_bytes(b'SYNTHETIC identity fixture; NOT a benchmark image'+(shift.encode() if different else b''))
                    writer.writerow({'shift':shift,'gt_class':2,'seed':seed,
                                     'local_path':'../outside.jpg' if escape else str(image.relative_to(self.root))})
        return path

    def test_actual_bytes_equality_required_not_seed_alone(self):
        good,audit=verify_local_baseline_manifest(self.manifest(),self.keys)
        self.assertEqual(good,{(2,1)})
        bad,audit=verify_local_baseline_manifest(self.manifest(different=True),self.keys)
        self.assertEqual(bad,set()); self.assertEqual(audit['mismatched_baseline_pairs'],1)

    def test_repeated_source_hash_alias_is_not_an_independent_sample(self):
        with self.assertRaises(SchemaError): verify_local_baseline_manifest(self.manifest(seeds=(1,2)),self.keys)

    def test_manifest_cannot_escape_evidence_directory(self):
        with self.assertRaises(SchemaError): verify_local_baseline_manifest(self.manifest(escape=True),self.keys)


if __name__=='__main__': unittest.main(verbosity=2)
