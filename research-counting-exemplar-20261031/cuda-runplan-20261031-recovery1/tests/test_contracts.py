import importlib.util
import unittest
from copy import deepcopy


class ContractTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('contracts'),'CUDA pipeline contracts missing')
        import contracts
        return contracts

    def test_compute_gate_covers_all_492_and_nonzero_io_reserve(self):
        m=self.module()
        self.assertTrue(m.compute_gate(100,0,[1.,.9,.8],60)['allowed'])
        self.assertFalse(m.compute_gate(100,0,[2.,.9,.8],60)['allowed'])
        self.assertEqual(m.compute_gate(100,0,[1.,.9,.8],60)['forecast_seconds'],1144)
        with self.assertRaises(ValueError):m.compute_gate(100,0,[1.,.9],60)
        with self.assertRaises(ValueError):m.compute_gate(100,0,[1.,.9,.8],0)

    def test_one_credit_guard_uses_projected_rate_and_rounding_buffer(self):
        m=self.module()
        self.assertTrue(m.credit_gate(1.15,0,2700,2)['allowed'])
        self.assertFalse(m.credit_gate(3.,0,2700,2)['allowed'])
        self.assertFalse(m.credit_gate(1.,1.96,1,2)['allowed'])
        self.assertTrue(m.credit_gate(2.45,0,2700,2)['allowed'])
        self.assertFalse(m.credit_gate(2.45,.14396,2700,2)['allowed'])

    def test_control_frame_ignores_but_does_not_parse_diagnostics(self):
        m=self.module()
        text='[colab] Pruned 1 stale local session(s).\nwarning from merged stderr\n'+m.CONTROL_PREFIX+'{"active_assignments":0,"assignments":[],"rate_per_hour":0}\ntrailing diagnostic\n'
        self.assertEqual(m.parse_control_frame(text)['active_assignments'],0)
        for bad in ('{"active_assignments":0}',m.CONTROL_PREFIX+'{bad}',text+text,m.CONTROL_PREFIX+'[]',m.CONTROL_PREFIX+'{} trailing'):
            with self.subTest(output=bad),self.assertRaises(ValueError):m.parse_control_frame(bad)

    def test_bad_artifact_path_and_hash_rejected_before_write(self):
        m=self.module()
        for path in ['/tmp/leak','../leak','x/../../leak']:
            with self.subTest(path=path),self.assertRaises(ValueError):m.validate_artifact_path(path)
        self.assertEqual(m.validate_artifact_path('results/query001.json'),'results/query001.json')
        data=b'hello';packet=m.make_packet('results/query001.json',data)
        self.assertEqual(m.decode_packet(packet),('results/query001.json',data))
        bad=packet.replace('aGVsbG8=','aGVsbGE=')
        with self.assertRaises(ValueError):m.decode_packet(bad)

    def test_missing_explicit_GO_never_allows_allocation(self):
        m=self.module()
        for auth in [None,{}, {'allocation_allowed':False},{'allocation_allowed':True,'freeze_sha256':'wrong'}]:
            with self.assertRaises(ValueError):m.validate_go(auth,'abc')
        m.validate_go({'allocation_allowed':True,'freeze_sha256':'abc','scope':'one_T4_precheck_then_conditional_492','max_allocations':1},'abc')

    def test_final_completion_requires_all_distinct_query_hashes(self):
        m=self.module();expected={'a':'h1','b':'h2'}
        self.assertTrue(m.complete_result_set([{'query_id':'a','query_sha256':'h1'},{'query_id':'b','query_sha256':'h2'}],expected))
        for rows in [[{'query_id':'a','query_sha256':'h1'}],[{'query_id':'a','query_sha256':'h1'}]*2,[{'query_id':'a','query_sha256':'h1'},{'query_id':'b','query_sha256':'bad'}]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):m.complete_result_set(rows,expected)

    def test_query_identity_rejects_lost_negative_group_or_A_B_swap(self):
        m=self.module();s={'filename':'fake.jpg','pair_id':'fake','A_boxes_xyxy':[[1,1,2,2]]*3,'B_boxes_xyxy':[[3,3,4,4]]*3,'A_count':5,'B_count':7,'selection_sha256':'rank'}
        label={'pair_id':'fake','eligible':True,'selected_images':[{'filename':'fake.jpg','A_count':5,'B_count':7}], 'frozen_input_texts':{'F_A':'red toy','F_B':'blue toy','C':'toy'}}
        q={'query_id':'fake.jpg/C/A','filename':'fake.jpg','pair_id':'fake','selection_rank_sha256':'rank','mode':'C','target':'A','text':'toy','negative_texts':[''],'exemplar_boxes':s['A_boxes_xyxy'],'negative_exemplar_boxes':[s['B_boxes_xyxy']],'target_count_from_official_annotation':5,'negative_count_from_official_annotation':7,'caption':'toy .  . ','image_sha256':'image'}
        import hashlib
        q['query_sha256']=hashlib.sha256(m.canonical(q)).hexdigest()
        m.validate_query(q,s,label)
        for key,value in [('negative_texts',[]),('exemplar_boxes',s['B_boxes_xyxy']),('target_count_from_official_annotation',7)]:
            bad=deepcopy(q);bad[key]=value;bad.pop('query_sha256');bad['query_sha256']=hashlib.sha256(m.canonical(bad)).hexdigest()
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_query(bad,s,label)
        bad_source=deepcopy(s);bad_source['A_count']=500
        with self.assertRaises(ValueError):m.validate_query(q,bad_source,label)
        bad_label=deepcopy(label);bad_label['selected_images'][0]['A_count']=500
        with self.assertRaises(ValueError):m.validate_query(q,s,bad_label)

    def test_summary_truth_is_bound_only_to_frozen_labels(self):
        m=self.module()
        label={'pair_id':'p','eligible':True,'reason':'', 'selected_images':[{'filename':'i1','A_count':1,'B_count':9},{'filename':'i2','A_count':1,'B_count':9}]}
        grouped={filename:{mode:{'A':2,'B':9} for mode in ('F','C','V')} for filename in ('i1','i2')}
        pairs,policy=m.bound_metric_inputs(grouped,[label])
        self.assertEqual(pairs[0]['images'][0]['truth'],{'A':1,'B':9})
        self.assertEqual(policy[0]['image_ids'],['i1','i2'])
        with self.assertRaises(ValueError):m.bound_metric_inputs(dict(grouped,extra={}),[label])


if __name__=='__main__':unittest.main(verbosity=2)
