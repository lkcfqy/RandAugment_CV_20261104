from pathlib import Path
from collections import Counter
from fractions import Fraction as F
from itertools import product
import ast, csv, hashlib, json, os, resource, time
START=time.perf_counter()
CPU=time.process_time()
os.nice(10)
resource.setrlimit(resource.RLIMIT_CPU, (3,3))
ROOT=Path("/Users/lkc/Downloads/RandAugment_CV_20261104")
OUT=Path(__file__).resolve().parent
COMMON=ROOT/"augmentation-research-20261022/pilot-20261022/common.py"
TV=ROOT/"causal-augmentation-research-20261020/area-pairing-cpu-20261021/encoder-readiness-20261021/runtime/.venv/lib/python3.14/site-packages/torchvision/transforms/autoaugment.py"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):
    with (OUT/name).open("x") as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")
def val(q): return {"fraction":str(q),"value":float(q)}
ct=ast.parse(COMMON.read_text())
calls=[n for n in ast.walk(ct) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="RandAugment"]
assert len(calls)==1
kwargs={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in ("num_ops","magnitude","num_magnitude_bins")}
assert kwargs==dict(num_ops=2,magnitude=9,num_magnitude_bins=31),kwargs
tree=ast.parse(TV.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="RandAugment")
space=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_augmentation_space")
ret=next(n for n in ast.walk(space) if isinstance(n,ast.Return))
ops=[ast.literal_eval(k) for k in ret.value.keys]
expected=["Identity","ShearX","ShearY","TranslateX","TranslateY","Rotate","Brightness","Color","Contrast","Sharpness","Posterize","Solarize","AutoContrast","Equalize"]
assert ops==expected
target="Posterize"; identity="Identity"; K=len(ops); N=kwargs["num_ops"]
remaining=[x for x in ops if x!=target]
base=Counter(); zero=Counter(); reroute=Counter(); direct=Counter()
branch_count=0
for pair in product(ops,repeat=N):
    mass=F(1,K**N); base[pair]+=mass
    z=tuple(identity if x==target else x for x in pair); zero[z]+=mass
    choices=[remaining if x==target else [x] for x in pair]
    b=1
    for c in choices:b*=len(c)
    for routed in product(*choices):
        reroute[routed]+=mass/b;branch_count+=1
for pair in product(remaining,repeat=N): direct[pair]+=F(1,len(remaining)**N)
assert reroute==direct
laws={"B":base,"Z":zero,"R":reroute}
for law in laws.values():assert sum(law.values())==1
marginal={}
for arm,law in laws.items():
    marginal[arm]=[]
    for slot in range(N):
        m={op:sum((p for seq,p in law.items() if seq[slot]==op),F()) for op in ops}
        marginal[arm].append(m)
for slot in range(N):
    for op in remaining:
        if op!=identity:assert marginal["B"][slot][op]==marginal["Z"][slot][op]==F(1,K)
    assert marginal["Z"][slot][identity]==F(2,K)
    assert all(marginal["R"][slot][op]==F(1,K-1) for op in remaining)
geo={"ShearX","ShearY","TranslateX","TranslateY","Rotate"}
summ={}
for arm,law in laws.items():
    summ[arm]={
      "unique_name_sequences":len(law),
      "expected_non_identity_name_count":val(sum((p*sum(op!=identity for op in seq) for seq,p in law.items()),F())),
      "p_at_least_one_geometry_name":val(sum((p for seq,p in law.items() if any(op in geo for op in seq)),F())),
      "p_at_least_one_rotate":val(sum((p for seq,p in law.items() if "Rotate" in seq),F())),
      "p_at_least_one_identity_name":val(sum((p for seq,p in law.items() if identity in seq),F()))
    }
tvs={}
for a,b in (("B","Z"),("B","R"),("Z","R")):
    names=set(laws[a])|set(laws[b])
    q=sum((abs(laws[a][x]-laws[b][x]) for x in names),F())/2
    tvs[a+"_"+b]=val(q)
trigger=sum((p for seq,p in base.items() if target in seq),F())
assert trigger==1-F(K-1,K)**N
assert tvs["B_Z"]["fraction"]==str(F(27,196))
assert tvs["Z_R"]["fraction"]==str(F(972,8281))
with (OUT/"pair-laws.csv").open("x",newline="") as f:
    w=csv.writer(f);w.writerow(["op1","op2","B_fraction","Z_fraction","R_fraction"])
    for seq in product(ops,repeat=N):w.writerow([*seq,*[str(laws[a][seq]) for a in ("B","Z","R")]])
result={
 "configuration":kwargs,"operation_names":ops,"target":target,
 "scope":"Exact name-sequence distributions, not pixel distributions, semantic changes or trained accuracy.",
 "B_sequences":K**N,"R_direct_sequences":len(remaining)**N,"coupled_replacement_branches":branch_count,
 "per_slot_non_identity_survivor_probability":{a:val(marginal[a][0]["Rotate"]) for a in laws},
 "survivor_relative_exposure_increase":val(F(K,K-1)-1),
 "two_surviving_names_relative_mass_increase":val(F(K*K,(K-1)**2)-1),
 "original_sequence_contains_target":val(trigger),
 "summaries":summ,"name_sequence_total_variation":tvs,
 "assertions_passed":True,
 "images_read":0,"model_forwards":0,"training_runs":0,"gpu_calls":0,"random_seed":None,
 "input_hashes":{
    str(COMMON.relative_to(ROOT)):sha(COMMON),str(TV.relative_to(ROOT)):sha(TV),
    "PROTOCOL_ZH.md":sha(OUT/"PROTOCOL_ZH.md"),"exact_law_check.py":sha(Path(__file__))
 },
 "source_lines":{"common_RandAugment_call":calls[0].lineno,"torchvision_RandAugment_class":cls.lineno,"augmentation_space":space.lineno},
 "worker_wall_seconds":time.perf_counter()-START,"worker_cpu_seconds":time.process_time()-CPU,
 "worker_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
}
dump("exact-law-result.json",result)
print(json.dumps({"assertions_passed":True,"branches":branch_count,"worker_wall_seconds":result["worker_wall_seconds"]}))
