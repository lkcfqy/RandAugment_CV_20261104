from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from math import comb
import csv,hashlib,json,os,resource,time
t=time.perf_counter();c=time.process_time()
os.nice(10);resource.setrlimit(resource.RLIMIT_CPU,(1,1))
ROOT=Path("/Users/lkc/Downloads/RandAugment_CV_20261104")
OUT=Path(__file__).resolve().parent
OLD=ROOT/"augmentation-research-20261022/nm-composition-screen-20261023"
rows=list(csv.DictReader((OLD/"pair-laws.csv").open()))
assert len(rows)==196
names=list(dict.fromkeys(row["op1"] for row in rows))
assert len(names)==14 and "Identity" in names and "Posterize" in names
survivors=[s for s in names if s!="Posterize"]
def compress(seq):return tuple(s for s in seq if s!="Identity")
z=Counter();r2=Counter();r1=Counter()
for row in rows:
    seq=compress((row["op1"],row["op2"]))
    z[seq]+=F(row["Z_fraction"])
    r2[seq]+=F(row["R_fraction"])
for op in survivors:r1[compress((op,))]+=F(1,13)
weights=[F(comb(2,d))*F(13,14)**d*F(1,14)**(2-d) for d in range(3)]
mix=Counter()
for seq,p in {():F(1)}.items():mix[seq]+=weights[0]*p
for seq,p in r1.items():mix[seq]+=weights[1]*p
for seq,p in r2.items():mix[seq]+=weights[2]*p
support={seq for seq in set(z)|set(mix) if z[seq]!=0 or mix[seq]!=0}
diff={seq:z[seq]-mix[seq] for seq in support}
assert all(v==0 for v in diff.values())
assert sum(z.values())==sum(mix.values())==1
assert weights==[F(1,196),F(26,196),F(169,196)]
result={"passed":True,"statement":"Z=(I/14+13Q/14)^2=I/196+26Q/196+169Q^2/196, where Q uniformly draws from the remaining13 including original Identity. Equality is of ordered operation kernels after removing neutral Identity steps; it is not equality of trained performance across Z and R=Q^2.",
"weights":[str(w) for w in weights],"positive_compressed_sequences":len(support),"maximum_absolute_fraction_difference":str(max(map(abs,diff.values()))),"seed":None,"source_pair_laws_sha256":hashlib.sha256((OLD/"pair-laws.csv").read_bytes()).hexdigest(),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"protocol_sha256":hashlib.sha256((OUT/"PROTOCOL_ZH.md").read_bytes()).hexdigest(),"images_read":0,"model_forwards":0,"training_runs":0,"gpu_calls":0,"worker_wall_seconds":time.perf_counter()-t,"worker_cpu_seconds":time.process_time()-c}
with (OUT/"depth-equivalence.json").open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps({"passed":True,"sequences":len(support),"difference":"0"}))
