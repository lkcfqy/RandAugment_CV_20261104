from pathlib import Path
from fractions import Fraction as F
import csv,hashlib,json,time,resource,os
start=time.perf_counter();cpu=time.process_time();os.nice(10)
ROOT=Path("/Users/lkc/Downloads/RandAugment_CV_20261104")
OUT=ROOT/"augmentation-research-20261022/nm-composition-screen-20261023"
r=json.loads((OUT/"exact-law-result.json").read_text())
locked=json.loads((OUT/"protocol-lock.json").read_text())
for name,h in locked.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h
for name,h in r["input_hashes"].items():
    p=OUT/name if name in ("PROTOCOL_ZH.md","exact_law_check.py") else ROOT/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==h
ledger=json.loads((OUT/"compute-ledger.json").read_text())
assert hashlib.sha256((ROOT/ledger["previous_ledger_path"]).read_bytes()).hexdigest()==ledger["previous_ledger_sha256"]
# Independent closed-form Bernoulli/slot probabilities, no regeneration of the enumeration.
K,N=14,2
closed={
"B":{"expected_non_identity_name_count":N*F(13,14),"p_at_least_one_geometry_name":1-F(9,14)**N,"p_at_least_one_rotate":1-F(13,14)**N,"p_at_least_one_identity_name":1-F(13,14)**N},
"Z":{"expected_non_identity_name_count":N*F(12,14),"p_at_least_one_geometry_name":1-F(9,14)**N,"p_at_least_one_rotate":1-F(13,14)**N,"p_at_least_one_identity_name":1-F(12,14)**N},
"R":{"expected_non_identity_name_count":N*F(12,13),"p_at_least_one_geometry_name":1-F(8,13)**N,"p_at_least_one_rotate":1-F(12,13)**N,"p_at_least_one_identity_name":1-F(12,13)**N}}
checks=0
for arm,v in closed.items():
 for key,q in v.items():
  assert F(r["summaries"][arm][key]["fraction"])==q
  checks+=1
assert F(r["name_sequence_total_variation"]["Z_R"]["fraction"])==F(12,13)**2-F(12,14)**2;checks+=1
assert F(r["original_sequence_contains_target"]["fraction"])==1-F(13,14)**2;checks+=1
rows=list(csv.DictReader((OUT/"pair-laws.csv").open()))
assert len(rows)==196
for arm in ("B","Z","R"):assert sum(F(row[arm+"_fraction"]) for row in rows)==1;checks+=1
for row in rows:
 a,b=row["op1"],row["op2"]
 assert F(row["B_fraction"])==F(1,196)
 expectedR=F(0) if "Posterize" in (a,b) else F(1,169)
 zfactor=(2 if a=="Identity" else 0 if a=="Posterize" else 1)*(2 if b=="Identity" else 0 if b=="Posterize" else 1)
 assert F(row["R_fraction"])==expectedR
 assert F(row["Z_fraction"])==F(zfactor,196)
 checks+=3
assert (OUT/"protocol-lock.json").stat().st_mtime_ns < (OUT/"exact-law-result.json").stat().st_mtime_ns
result={"passed":True,"independent_formula_checks":checks,"input_hashes_unchanged":True,"previous_ledger_unchanged":True,"freeze_predates_result":True,"model_forwards":0,"image_decodes":0,"enumeration_reruns":0,"verification_worker_wall_seconds":time.perf_counter()-start,"verification_worker_cpu_seconds":time.process_time()-cpu}
with (OUT/"verification.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
print(json.dumps(result))
