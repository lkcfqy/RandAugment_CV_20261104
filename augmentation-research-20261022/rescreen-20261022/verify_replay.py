"""Separate standard-library recomputation from the saved per-photo CSV."""
import csv, hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'controller-temperature-per-photo.csv').open()))
result=json.loads((ROOT/'controller-temperature-result.json').read_text())
def quantile(a,q):
    a=sorted(a); t=(len(a)-1)*q; i=math.floor(t); j=math.ceil(t)
    return a[i]+(a[j]-a[i])*(t-i)
delta=[abs(float(r['MIS_T1'])-float(r['MIS_T2'])) for r in rows]
count=sum(abs(int(r['rank_T1'])-int(r['rank_T2']))/(len(rows)-1)>=.1 for r in rows)
p95=quantile(delta,.95)
assert len(rows)==589 and count/len(rows)<.1
assert math.isclose(p95,result['contrasts']['1_vs_2']['raw_MIS_absolute_change_quantiles']['0.95'],rel_tol=0,abs_tol=1e-14)
base=sorted(float(r['MIS_T1']) for r in rows)
for t in ['0.5','1','2']:
    assert sorted(float(r[f'matched_MIS_T{t}']) for r in rows)==base
    assert sorted(int(r[f'rank_T{t}']) for r in rows)==list(range(len(rows)))
for p,digest in result['input_hashes_before_and_after_equal'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
checks={'status':'passed_separate_standard_library_recomputation',
        'independent_agent_review':False,
        'n':len(rows),'main_rank_decile_count':count,'main_rank_decile_fraction':count/len(rows),
        'main_raw_delta_p95':p95,'gate':False,
        'all_exact_sorted_multisets_match':True,'source_and_plan_and_code_hashes_match':True}
(ROOT/'replay-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
