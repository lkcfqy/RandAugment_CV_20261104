"""Resolve pre-result duplicate decisions without replacement or outcome access."""
from common import *
from datetime import datetime,timezone

def main():
    assert not (ROOT/'results').exists()
    qa=json.loads((ROOT/'qa/source-qa.json').read_text())
    assert not qa['issues'],'Resolve orientation/integrity issues before freezing.'
    decisions=json.loads((ROOT/'qa/duplicate-adjudication.json').read_text())
    expected={tuple(sorted((p['a'],p['b']))) for p in qa['dhash_distance4_candidate_pairs']}
    actual={tuple(sorted((p['a'],p['b']))) for p in decisions['pairs']}
    assert expected==actual and len(actual)==len(decisions['pairs'])
    rows=qa['rows'];parent={r['source_id']:r['source_id'] for r in rows}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    def union(a,b):parent[find(b)]=find(a)
    grouped=set()
    for group in qa['exact_decoded_duplicate_groups']:
        for sid in group:grouped.add(sid);union(group[0],sid)
    for pair in decisions['pairs']:
        assert pair['decision'] in {'same_capture','uncertain','different'}
        if pair['decision']!='different':
            union(pair['a'],pair['b']);grouped.update([pair['a'],pair['b']])
    clusters={}
    for sid in sorted(grouped):clusters.setdefault(find(sid),[]).append(sid)
    kept=[];excluded=[]
    for row in rows:
        r=dict(row)
        if r['source_id'] in grouped:
            r['exclusion_reason']='Entire confirmed/uncertain duplicate cluster removed before fitting; no replacement.'
            r['cluster_members']=clusters[find(r['source_id'])];excluded.append(r)
        else:
            r['duplicate_cluster_id']='unique-source:'+r['source_id'];kept.append(r)
    manifest={'status':'frozen_before_task_fitting','at_utc':datetime.now(timezone.utc).isoformat(),
        'class_names':qa['class_names'],'rows':kept,'excluded_rows':excluded,
        'duplicate_clusters_removed':list(clusters.values()),'replacement_samples':0,
        'source_QA_sha256':sha(ROOT/'qa/source-qa.json'),
        'duplicate_decisions_sha256':sha(ROOT/'qa/duplicate-adjudication.json'),
        'candidate_plan_sha256':sha(PLAN_PATH),'reserved_members_extracted_or_decoded':False,
        'animal_identity_and_pretraining_overlap_verified_absent':False}
    validated_rows(manifest)
    dump(ROOT/'fit-manifest.json',manifest)
    print(json.dumps({'status':'fit_manifest_frozen','kept':len(kept),'excluded':len(excluded),
        'train':sum(r['candidate_partition']=='train' for r in kept),
        'development':sum(r['candidate_partition']=='development' for r in kept),
        'clusters_removed':list(clusters.values())}),flush=True)

if __name__=='__main__':main()
