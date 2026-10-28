from pathlib import Path
import hashlib,json,os,subprocess,sys,time,resource
ROOT=Path("/Users/lkc/Downloads/RandAugment_CV_20261104")
OUT=Path(__file__).resolve().parent
ledger_path=ROOT/"augmentation-research-20261022/ra-dev-paired-cpu-20261023/compute-ledger.json"
old_bytes=ledger_path.read_bytes(); old=json.loads(old_bytes)
assert old["paired_stages_cumulative_child_inclusive_seconds"]==22.682998540927656
assert old["paired_stages_cumulative_forwards"]==212
env=os.environ.copy()
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):env[k]="1"
env["PYTHONDONTWRITEBYTECODE"]="1"
started=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
status="uncertain";code=None
try:
    p=subprocess.run([sys.executable,"-B",str(OUT/"exact_law_check.py")],cwd=ROOT,env=env,capture_output=True,text=True,timeout=5)
    status="completed" if p.returncode==0 else "failed"
    code=p.returncode
    stdout=p.stdout;stderr=p.stderr
except subprocess.TimeoutExpired as e:
    status="timeout_stopped";stdout=str(e.stdout or "");stderr=str(e.stderr or "")
elapsed=time.perf_counter()-started;after=resource.getrusage(resource.RUSAGE_CHILDREN)
assert ledger_path.read_bytes()==old_bytes
record={
 "stage":"nm-composition-screen-20261023","status":status,"exit_code":code,
 "previous_ledger_path":str(ledger_path.relative_to(ROOT)),
 "previous_ledger_sha256":hashlib.sha256(old_bytes).hexdigest(),
 "previous_paired_stages_child_inclusive_seconds":old["paired_stages_cumulative_child_inclusive_seconds"],
 "previous_paired_stages_forwards":212,
 "new_exact_check_child_inclusive_seconds":elapsed,
 "new_exact_check_child_cpu_seconds":after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
 "new_model_forwards":0,"new_training_runs":0,"new_image_decodes":0,"new_gpu_calls":0,
 "recorded_paired_plus_exact_check_child_seconds":old["paired_stages_cumulative_child_inclusive_seconds"]+elapsed,
 "recorded_paired_plus_exact_check_forwards":212,
 "scope":"Two paired probes plus this exact-law check. Earlier training/cache generation and historical audits stay in their original records; not a total project lifetime compute claim.",
 "wall_budget_seconds":5,"cpu_budget_seconds":3,"no_automatic_retry":True
}
for name,data in (("compute-ledger.json",record),("execution.json",{"stdout":stdout,"stderr":stderr,"status":status,"exit_code":code})):
    with (OUT/name).open("x") as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps(record))
