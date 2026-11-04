"""Post-run numerical verification from saved caches; no encoder, no fitting."""
from common import *
import datetime
m=json.loads((ROOT/"IDENTITIES.json").read_text())
p=json.loads((ROOT/"PROTOCOL.json").read_text())
seal=json.loads((ROOT/"SELECTION_SEAL.json").read_text())
heads=json.loads((ROOT/"HEAD_IDENTITIES.json").read_text())
run=json.loads((ROOT/"RUN_RECEIPT.json").read_text())
assert run["status"]=="COMPLETE"
t=time.perf_counter()
def cache(role):
    idx=json.loads((ROOT/(role.upper()+"_CACHE_INDEX.json")).read_text())
    arrays=[];names=[]
    for x in idx:
        assert sha(ROOT/x["path"])==x["sha256"]
        with np.load(ROOT/x["path"],allow_pickle=False) as d:
            arrays.append(d["features"]);names.extend(d["member"].tolist())
    return np.concatenate(arrays),names
s,snames=cache("support");q,qnames=cache("query")
assert snames==[r["member"] for r in m["support_union"]]
query=json.loads((ROOT/"QUERY_IDENTITIES.json").read_text())
assert qnames==[r["member"] for r in query]
assert not {x["rgb_sha256"] for x in query}&{x["rgb_sha256"] for x in m["support_union"]}
for row in m["support_union"]+query:assert m["classes"][row["label"]]==row["class"]
assert np.isfinite(s).all() and np.isfinite(q).all()
normerr=max(float(np.max(np.abs(np.linalg.norm(s,axis=-1)-1))),float(np.max(np.abs(np.linalg.norm(q,axis=-1)-1))))
assert normerr<1e-6
with np.load(ROOT/"HEADS.npz",allow_pickle=False) as d:Ws=d["W"];bs=d["b"]
with np.load(ROOT/"QUERY_PREDICTIONS.npz",allow_pickle=False) as d:qs=d["scores"];qp=d["prediction"]
reproject=np.stack([scores(q[:,0],W,b) for W,b in zip(Ws,bs)])
scoreerr=float(np.max(np.abs(reproject-qs)))
assert scoreerr==0 and np.array_equal(qp,reproject.argmax(-1))
lookup={n:i for i,n in enumerate(snames)};episodes={(e["episode"],e["B"]):e for e in m["episodes"]}
maxg=maxb=0.
for h,W,b in zip(heads,Ws,bs):
    rows=episodes[h["episode"],h["B"]]["support"]
    rows=[r for r in rows if h["fold"]==-1 or r["fold"]!=h["fold"]]
    inds=[lookup[r["member"]] for r in rows]
    X=policy_features(s[inds],h["policy"]).astype(np.float64).reshape(-1,384)
    Y=np.eye(10)[np.repeat([r["label"] for r in rows],4)]
    residual=X@W+b-Y
    maxg=max(maxg,float(np.max(np.abs(X.T@residual/len(X)+LAMBDA*W))))
    maxb=max(maxb,float(np.max(np.abs(residual.mean(0)))))
assert maxg<1e-10 and maxb<1e-10
with np.load(ROOT/"CV_PREDICTIONS.npz",allow_pickle=False) as d:
    hi=d["head_index"];ci=d["support_union_index"];vs=d["scores"];vp=d["prediction"];vy=d["label"]
    cv_re=np.stack([scores(s[i,0],Ws[h],bs[h]) for h,i in zip(hi,ci)])
    cverr=float(np.max(np.abs(vs-cv_re)))
    assert cverr<1e-12 and np.array_equal(vp,cv_re.argmax(-1))
    for h,i,y in zip(hi,ci,vy):
        info=heads[int(h)];row=[r for r in episodes[info["episode"],info["B"]]["support"] if r["member"]==snames[int(i)]]
        assert len(row)==1 and row[0]["fold"]==info["fold"] and row[0]["label"]==int(y)
freeze=json.loads((ROOT/"CODE_FREEZE.json").read_text())
for name,h in freeze["code"].items():assert sha(ROOT/name)==h
assert run["initial_state_hash"]==run["final_state_hash"] and run["query_start_utc"]>seal["utc"]
result={"status":"PASS","verified_real_heads":360,"query_score_entries":int(qs.size),"query_predictions":int(qp.size),
"cv_original_predictions":int(len(vp)),"max_query_score_reprojection_abs_error":scoreerr,"max_cv_score_reprojection_abs_error":cverr,
"max_ridge_weight_normal_equation_abs_residual":maxg,"max_ridge_intercept_normal_equation_abs_residual":maxb,
"max_feature_unit_norm_error":normerr,"all_cv_validation_originals_excluded_from_their_fit":True,"all_cache_file_hashes_verified":True,
"encoder_state_unchanged":True,"frozen_core_code_unchanged":True,"seconds":time.perf_counter()-t,
"new_model_forwards":0,"new_real_head_fits":0,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
atomic_json("POSTRUN_VERIFICATION.json",result);print(json.dumps(result,indent=2))

