"""No-model checks for deterministic selection, unchanged computation and persistence."""
import ast,collections,copy,hashlib,json,math,pathlib,tempfile,unittest
import diagnostic_contracts as c

class Tests(unittest.TestCase):
    def test_selection_indices_independent(self):
        s=json.loads((c.P/'SELECTION.json').read_text())
        self.assertEqual(len(s['groups']),29)
        for g in s['groups']:
            want=[math.floor((j+.5)*g['n']/g['m']) for j in range(g['m'])]
            self.assertEqual(want,g['indices']);self.assertEqual(len(set(want)),g['m'])
        self.assertEqual(len(s['image_ids']),232);self.assertEqual(s['image_ids'],sorted(set(s['image_ids'])))

    def test_all_groups_and_full_annotations(self):
        s=json.loads((c.P/'SELECTION.json').read_text());old=json.loads((c.R/'confirmation-pool-plan-20261101/confirmation_annotations.json').read_text());new=json.loads((c.P/'data/annotations.json').read_text())
        ids=set(s['image_ids'])
        self.assertEqual(new['annotations'],[a for a in old['annotations'] if a['image_id'] in ids])
        self.assertEqual(len(new['annotations']),576)
        strata=json.loads((c.P/'STRATA.json').read_text())['strata']
        flat=[x for v in strata.values() for x in v]
        self.assertEqual(collections.Counter(flat),collections.Counter(ids))

    def test_identity_matrix(self):
        models=[{'seed':20261031,'arm':a} for a in ['P0','A0','A1']]
        ids=json.loads((c.P/'SELECTION.json').read_text())['image_ids']
        expected=c.expected_ids(models,ids);self.assertEqual(len(expected),696)
        c.require_complete_ids(list(reversed(expected)),expected)
        for actual in [expected[:-1],expected+[expected[0]]]:
            with self.assertRaises(c.ContractError):c.require_complete_ids(actual,expected)
        with self.assertRaises(c.ContractError):c.identity(20261031,'M','c100',ids[0])

    def test_metric_mapping(self):
        m=c.mapped_stats([.1]*12)
        self.assertEqual(m['AP']['maxDets'],100)
        for key in ['AP50','AP75','AP_small','AP_medium','AP_large']:self.assertEqual(m[key]['maxDets'],1000)
        for n in [100,300,1000]:self.assertEqual(m['AR'+str(n)]['maxDets'],n)

    def test_atomic_idempotent_and_conflict(self):
        with tempfile.TemporaryDirectory(dir=c.P) as tmp:
            p=pathlib.Path(tmp)/'fixture.json';value={'only':'synthetic'}
            self.assertFalse(c.atomic_json(p,value,new=True)['existing_exact'])
            self.assertTrue(c.atomic_json(p,value,new=True)['existing_exact'])
            with self.assertRaises(c.ContractError):c.atomic_json(p,{'other':'synthetic'},new=True)
            self.assertEqual(json.loads(p.read_text()),value)

    def test_payload_tamper_and_no_repeat(self):
        base=c.P
        try:
            with tempfile.TemporaryDirectory(dir=base) as tmp:
                c.P=pathlib.Path(tmp);(c.P/'predictions').mkdir()
                key=c.identity(20261031,'A0','c100',999)
                pin={'seed':20261031,'arm':'A0','condition':'c100','factor':1.,'image_id':999,'checkpoint_sha256':'dummy','config_sha256':'dummy','image_sha256':'dummy'}
                run={'identity_map':{key:pin},'dev_annotation':{'sha256':'dummy'},'environment_lock_sha256':'dummy'}
                payload={**pin,'identity':key,'annotation_sha256':'dummy','environment_lock_sha256':'dummy','predictions':[{'image_id':999,'category_id':1,'bbox':[1.,2.,3.,4.],'score':.5} for _ in range(300)]}
                self.assertFalse(c.persist_prediction(payload,run)['existing_exact'])
                self.assertTrue(c.persist_prediction(payload,run)['existing_exact'])
                path=c.P/'predictions'/(key+'.json');self.assertEqual(c.load_prediction(path,run),payload)
                tamper=json.loads(path.read_text());tamper['payload']['predictions'][0]['score']=.6;path.write_text(json.dumps(tamper))
                with self.assertRaises(c.ContractError):c.load_prediction(path,run)
        finally:c.P=base

    def test_original_safe_loaders_unchanged(self):
        new=ast.parse((c.P/'formal_cpu.py').read_text())
        for path,newname in [(c.R/'unadapted-reference-20261101/formal_cpu.py','safe_model_P0'),(c.R/'cpu-paired-AM-20261101/evaluation/formal_cpu.py','safe_model_A')]:
            old=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='safe_model')
            actual=copy.deepcopy(next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==newname));actual.name='safe_model'
            self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(actual,include_attributes=False))

    def test_original_forward_and_postprocess_unchanged(self):
        def core(text):return text[text.index('                f0=time.monotonic()'):text.index('                format_seconds=time.monotonic()-f1')]
        original=(c.R/'unadapted-reference-20261101/formal_cpu.py').read_text()
        self.assertEqual(core(original),core((c.P/'formal_cpu.py').read_text()))
        for f in ['formal_cpu.py','diagnostic_contracts.py','lifecycle.py','prepare_geometry.py']:
            ast.parse((c.P/f).read_text())

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    c.atomic_json(c.P/'TEST_RESULTS.json',{'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'models_imported_or_called':0},new=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
