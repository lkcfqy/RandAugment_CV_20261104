from counting_assets import ROOT, dump
import json,ast,random,hashlib,csv,collections
from tokenizers import Tokenizer
r=ROOT/'sources/pairtally/dataset/pairtally_dataset/annotations'
a=json.loads((r/'pairtally_annotations_simple.json').read_text());m=json.loads((r/'image_metadata.json').read_text());b=json.loads((r/'bbx_anno_valid.json').read_text());lookup={v['original_filename']:k for k,v in m.items()}
rng=random.Random(42);diff=[]
for old,x in b.items():
    k=lookup[old];maxv=max(v for field in ['pos','neg'] for box in x[field] for v in box)
    bound=512 if maxv<=512 else 1024 if maxv<=1024 else int(maxv*1.2)
    for field,countfield in [('points','positive_count'),('negative_points','negative_count')]:
        points=[[rng.uniform(0,bound),rng.uniform(0,bound)] for _ in range(m[k][countfield])]
        if a[k][field]!=points:diff.append({'filename':k,'field':field})
dump('reports/point_semantics.json',{'source':'sources/pairtally/dataset/tools/convert_cvat_to_pairtally.py','source_lines':{'random_generator':785,'bounds_inferred_from_six_boxes':810,'random_point_assignment':894,'default_seed':1079},'scientific_meaning':'The points and negative_points are synthetic uniform random placeholders, not measured object locations. Only list lengths encode the official class counts.','independent_seed42_reconstruction_matches_all_points':not diff,'mismatches':diff,'images_checked':len(a),'localization_ground_truth_available':False,'actual_exemplar_boxes':'From CVAT pos/neg xyxy, preserved unchanged and verified against main corner annotations.','not_a_model_failure':True})

# Parse the commented author-provided short-name taxonomy as data; never execute source.
source=(ROOT/'sources/pairtally/dataset/tools/convert_cvat_to_pairtally.py').read_text();start=source.index('# {\n');end=source.index('\n# }',start)+len('\n# }');taxonomy=ast.literal_eval('\n'.join(line[2:] for line in source[start:end].splitlines()))
pairs=json.loads((ROOT/'reports/pair_inventory.json').read_text());candidates=[]
for p in pairs:
    if p['test_type']!='INTRA':continue
    supercat,_,codeA,codeB=p['pair_id'].split('_');base=codeA[:-1];assert base==codeB[:-1]
    common=taxonomy[supercat][base][0];A,B=p['source_prompt_pairs'][0]
    x={'pair_id':p['pair_id'],'image_count':p['image_count'],'common_class_source':common,'common_class_candidate':common,'F_A_source':A,'F_B_source':B,'F_A_candidate':A,'F_B_candidate':B,'status':'candidate_pending_protocol_freeze','issues':[]}
    if p['pair_id']=='OTR_INTRA_BEA1_BEA2':
        x.update(F_A_candidate='blue or purple bead',F_B_candidate='orange or pink bead');x['issues'].append('Source labels omit object noun; original filenames use green/pink but exemplar pixels match blue-purple/orange-pink source groups. Add noun only after protocol freeze.')
    if p['pair_id']=='OTR_INTRA_BUT1_BUT2':
        x.update(F_A_candidate='clear button',F_B_candidate='opaque white or beige button',status='blocked_label_to_visual_group_conflict');x['issues'].append('Both selected images: all 3 A boxes show clear/transparent buttons and all 3 B boxes show opaque light buttons, opposite source prompts Beige / Clear. Source counts and A/B box identities unchanged. Requires independent semantic/count-group resolution before F can be called correct.')
    if p['pair_id']=='HOU_INTRA_HCP1_HCP2':
        x.update(common_class_candidate='hair clip',F_A_candidate='black hair clip',F_B_candidate='brown hair clip');x['issues'].append('Pixels show hair clips, not electric clippers; author name hair clipper appears lexical. Correction is a candidate only.')
    if p['pair_id']=='HOU_INTRA_TPK1_TPK2':
        x.update(common_class_candidate='dental pick',F_B_candidate='dental floss pick');x['issues'].append('Author common name toothpick does not precisely cover floss picks; proposed dental pick needs semantic protocol review.')
    if p['pair_id']=='OTR_INTRA_STO1_STO2':
        x['F_B_candidate']='yellow stone';x['issues'].append('Source yellowstone is concatenated; candidate correction only.')
    candidates.append(x)
dump('reports/common_class_candidates.json',{'status':'not frozen; no inference results used','source_taxonomy':'Author commented short-description taxonomy, converter lines 321-389. Ordinary object category; not FOO/FUN/HOU/OFF/OTR supercategory labels.','pairs':candidates})
with (ROOT/'reports/common_class_candidates.csv').open('w',newline='') as f:
    fields=['pair_id','image_count','common_class_source','common_class_candidate','F_A_source','F_B_source','F_A_candidate','F_B_candidate','status','issues'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({**x,'issues':' '.join(x['issues'])} for x in candidates)
md=['共同类别名候选表（待协议冻结；未跑模型）','', '| Pair | 图数 | C 共同类别候选 | F-A / F-B 原文 | 需复核 |','|---|---:|---|---|---|']
for x in candidates:md.append('| '+x['pair_id']+' | '+str(x['image_count'])+' | '+x['common_class_candidate']+' | '+x['F_A_source']+' / '+x['F_B_source']+' | '+(' '.join(x['issues']) or '—')+' |')
(ROOT/'reports/COMMON_CLASS_CANDIDATES_ZH.md').write_text('\n'.join(md)+'\n')

# CPU tokenizer only; reproduce native insertion control flow as plain list indexing.
# Never import or call CountGD++ add_exemplar_tokens/forward.
tokenizer=Tokenizer.from_file(str(ROOT/'models/bert-base-uncased/tokenizer.json'))
special={tokenizer.token_to_id(t) for t in ['[CLS]','[SEP]','.','?']};star=tokenizer.token_to_id('*')
assert star==1008 and tokenizer.token_to_id('.')==1012
def insert(ids,label):
    label_count=-1;insert_at=None
    for token_ind in range(len(ids)):
        if token_ind>0 and ids[token_ind-1] in special:label_count+=1
        if label_count==label:
            insert_at=token_ind
            while ids[insert_at] not in special:insert_at+=1
            break
    if label_count==-1:insert_at=1
    assert insert_at is not None
    return ids[:insert_at]+[star]*3+ids[insert_at:]
traces=[]
for mode,text,neg in [('V','',''),('C','button',''),('F','clear button','opaque white or beige button')]:
    caption=text+' . '+neg+' . ';encoded=tokenizer.encode(caption).ids
    afterA=insert(encoded,0);afterB=insert(afterA,1)
    traces.append({'mode':mode,'caption':caption,'before':encoded,'after_positive_group':afterA,'after_both_groups':afterB,'tokens':[tokenizer.id_to_token(i) for i in afterB]})
assert traces[0]['after_both_groups']==[101,1008,1008,1008,1012,1008,1008,1008,1012,102]
dump('reports/native_prompt_static_trace.json',{'method':'Actual standard Rust tokenizer JSON parse + plain Python list reproduction of inspected insertion loop; no repository Python code or learned model executed.','tokenizers_version':__import__('importlib.metadata',fromlist=['version']).version('tokenizers'),'negative_texts_required':[''],'negative_exemplar_groups_required':1,'positive_boxes_required':3,'negative_boxes_in_group_required':3,'traces':traces,'model_forward_calls':0,'native_runtime_smoke_passed':False,'note':'C example is illustrative and does not freeze label table; next authorized runtime must assert actual returned token sequence.'})
print(json.dumps({'random_points_all_match':not diff,'mismatch_count':len(diff),'label_candidates':len(candidates),'V_static_token_trace':'pass','semantic_blocked_pairs':[x['pair_id'] for x in candidates if x['status'].startswith('blocked')]}))
