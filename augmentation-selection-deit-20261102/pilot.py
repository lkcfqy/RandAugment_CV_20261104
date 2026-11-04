"""Run the full frozen protocol, with no query utility before all choices are sealed."""
from common import *
import zipfile,datetime,inspect,sys,traceback
import torchvision,transformers,safetensors,PIL

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def environment():
    files={"randaugment":inspect.getfile(T.RandAugment),"transforms":inspect.getfile(T.RandomResizedCrop)}
    timm=timm_module()
    from timm.models import deit,vision_transformer
    files["deit"]=inspect.getfile(deit)
    files["vision_transformer"]=inspect.getfile(vision_transformer)
    return {"python":sys.version,"packages":{n:m.__version__ for n,m in [("torch",torch),("torchvision",torchvision),("transformers",transformers),("numpy",np),("safetensors",safetensors),("Pillow",PIL),("timm",timm)]},
    "sources":{n:{"path":p,"sha256":sha(p)} for n,p in files.items()},"device":"cpu","threads":torch.get_num_threads(),"interop":torch.get_num_interop_threads(),"deterministic":torch.are_deterministic_algorithms_enabled(),"mps_used":False}

def extract(model,z,entries,role,protocol,ledger):
    allfeatures=[];index=[]
    width=9 if role=="support" else 1
    chunk_size=16 if role=="support" else 64
    phase_start=time.perf_counter()
    for offset in range(0,len(entries),chunk_size):
        rss,used=guard(protocol);batch=entries[offset:offset+chunk_size]
        tensors=[];t=time.perf_counter()
        for row in batch: tensors.extend(make_views(read_image(z,row),row["rgb_sha256"],all_views=role=="support"))
        transform_seconds=time.perf_counter()-t
        t=time.perf_counter();features=encode(model,tensors).reshape(len(batch),width,FEATURE_DIM)
        forward_seconds=time.perf_counter()-t
        ledger["encoder_forward_images"]+=len(tensors)
        t=time.perf_counter()
        path=f"cache/{role}_{offset:04d}.npz"
        atomic_npz(path,features=features,member=np.array([x["member"] for x in batch]))
        save_seconds=time.perf_counter()-t
        index.append({"path":path,"offset":offset,"originals":len(batch),"views":width,"sha256":sha(ROOT/path),"transform_seconds":transform_seconds,"forward_seconds":forward_seconds,"save_seconds":save_seconds,"rss_bytes":rss})
        allfeatures.append(features)
        atomic_json(f"{role.upper()}_CACHE_INDEX.json",index)
        atomic_json("STATUS.json",{"status":"RUNNING","phase":role+"_features","originals_complete":offset+len(batch),"originals_total":len(entries),"encoder_forward_images":ledger["encoder_forward_images"],"query_scores_computed":False,"utc":utc()})
        if offset==0 or (offset//chunk_size)%8==0 or offset+len(batch)==len(entries):
            print(json.dumps({"phase":role+"_features","complete":offset+len(batch),"total":len(entries),"elapsed_seconds":time.perf_counter()-phase_start}),flush=True)
    ledger[role+"_feature_seconds"]=time.perf_counter()-phase_start
    return np.concatenate(allfeatures,axis=0)

def main():
    protocol=json.loads((ROOT/"PROTOCOL.json").read_text());m=json.loads((ROOT/"IDENTITIES.json").read_text())
    assert sha(ROOT/"IDENTITIES.json")==protocol["identities_sha256"]
    assert json.loads((ROOT/"TEST_RESULTS.json").read_text())["status"]=="PASS"
    contract=json.loads((ROOT/"SECOND_ENCODER_CONTRACT_TEST.json").read_text())
    assert contract["status"]=="PASS" and contract["weak_ra_bitwise_equal_views"]==6976
    assert sha(ZIP)==protocol["zip_sha256"]
    assert not (ROOT/"CODE_FREEZE.json").exists(),"do not duplicate an existing run"
    assert sha(MODEL)==protocol["encoder_weight_sha256"]
    code={x.name:sha(x) for x in sorted(ROOT.glob("*.py"))}
    atomic_json("CODE_FREEZE.json",{"utc":utc(),"protocol_sha256":sha(ROOT/"PROTOCOL.json"),"code":code,"environment":environment(),"query_utility_observed":False})
    ledger={"started_utc":utc(),"encoder_forward_images":0,"head_fits":0,"query_scores_computed":False,"new_CU":0}
    start=time.perf_counter();phase="load"
    try:
        guard(protocol);t=time.perf_counter();model=load_model();ledger["model_load_seconds"]=time.perf_counter()-t
        ledger["initial_state_hash"]=state_hash(model)
        with zipfile.ZipFile(ZIP) as z:
            phase="preflight"
            entries=sorted(m["support_union"],key=lambda r:key("timing",r["rgb_sha256"]))[:16]
            t=time.perf_counter();tensors=[]
            for row in entries:tensors.extend(make_views(read_image(z,row),row["rgb_sha256"]))
            transform_per_original=(time.perf_counter()-t)/16
            encode(model,tensors[:16]);ledger["encoder_forward_images"]+=16
            timings=[]
            for i in range(3):
                t=time.perf_counter();f=encode(model,tensors[16*(i+1):16*(i+2)]);timings.append(time.perf_counter()-t)
                assert f.shape==(16,FEATURE_DIM) and np.isfinite(f).all()
                ledger["encoder_forward_images"]+=16
            rng=np.random.default_rng(20261102)
            fake=rng.normal(size=(160,4,FEATURE_DIM));fake/=np.linalg.norm(fake,axis=-1,keepdims=True)
            times=[]
            for _ in range(3):
                t=time.perf_counter();ridge(fake,np.arange(160)%10);times.append(time.perf_counter()-t)
            remaining=protocol["deadline_epoch"]-time.time()
            per_image=max(timings)/16
            options=[]
            for qpc in [100]:
                support_cost=len(m["support_union"])*(9*per_image+transform_per_original)
                query_cost=qpc*10*(per_image+transform_per_original)
                heads_cost=360*max(times)
                total=1.5*(support_cost+query_cost+heads_cost+120)+300
                options.append({"query_per_class":qpc,"support_feature_seconds":support_cost,"query_feature_seconds":query_cost,"head_fit_seconds":heads_cost,"save_score_allowance_seconds":120,"multiplier":1.5,"archive_reserve_seconds":300,"predicted_remaining_seconds":total,"fits":total<remaining})
            acceptable=[x for x in options if x["fits"]]
            if not acceptable:raise RuntimeError("no complete-run cost option fits remaining time")
            selected=acceptable[0];qpc=selected["query_per_class"]
            query=[row for c in range(10) for row in [r for r in m["query"] if r["label"]==c][:qpc]]
            atomic_json("EXECUTION_PLAN.json",{"utc":utc(),"query_utility_observed":False,"encoder_batch":16,"encoder_timed_batch_seconds":timings,"transform_seconds_per_original":transform_per_original,"synthetic_head_fit_seconds":times,"options":options,"selected":selected,"query_members":[r["member"] for r in query],"query_reduced":qpc!=100,"remaining_wall_seconds_at_gate":remaining,"preflight_encoder_images":64,"preflight_has_real_label_predictions":False})
            print(json.dumps({"phase":"preflight_pass","query_images":len(query),"predicted_remaining_minutes":selected["predicted_remaining_seconds"]/60,"available_minutes":remaining/60}),flush=True)
            del tensors,f,fake
            phase="support"
            support_features=extract(model,z,m["support_union"],"support",protocol,ledger)
            assert support_features.shape==(len(m["support_union"]),9,FEATURE_DIM)
            member_index={r["member"]:i for i,r in enumerate(m["support_union"])}
            phase="fit";tfit=time.perf_counter()
            Wlist=[];blist=[];heads=[];choices=[];vc_head=[];vc_member=[];vc_y=[];vc_pred=[];vc_scores=[]
            for ep in m["episodes"]:
                rows=ep["support"];inds=np.array([member_index[r["member"]] for r in rows])
                y=np.array([r["label"] for r in rows]);folds=np.array([r["fold"] for r in rows])
                cached=support_features[inds];correct=[];full_head_indices=[]
                for p in POLICIES:
                    x=policy_features(cached,p);c=0
                    for fold in [0,1,2,3,-1]:
                        guard(protocol);train=np.ones(len(y),dtype=bool) if fold==-1 else folds!=fold
                        t=time.perf_counter();W,b=ridge(x[train],y[train]);duration=time.perf_counter()-t
                        hi=len(heads)
                        heads.append({"index":hi,"episode":ep["episode"],"B":ep["B"],"policy":p,"fold":fold,"train_originals":int(train.sum()),"views_per_original":4,"fit_seconds":duration})
                        Wlist.append(W);blist.append(b);ledger["head_fits"]+=1
                        if fold==-1:full_head_indices.append(hi)
                        else:
                            val=~train;sv=scores(cached[val,0],W,b);pred=sv.argmax(1);c+=int((pred==y[val]).sum())
                            vc_head.extend([hi]*int(val.sum()));vc_member.extend(inds[val].tolist());vc_y.extend(y[val].tolist());vc_pred.extend(pred.tolist());vc_scores.extend(sv)
                    correct.append(c)
                winner=choose(correct)
                choices.append({"episode":ep["episode"],"B":ep["B"],"cv_correct":correct,"cv_total":len(y),"selected_index":winner,"selected_policy":POLICIES[winner],"full_head_indices":full_head_indices})
                print(json.dumps({"phase":"head_fits","complete":ledger["head_fits"],"total":360,"query_utility_observed":False}),flush=True)
            assert ledger["head_fits"]==360 and len(choices)==24
            ledger["all_head_fit_seconds"]=time.perf_counter()-tfit
            atomic_npz("HEADS.npz",W=np.stack(Wlist),b=np.stack(blist))
            atomic_json("HEAD_IDENTITIES.json",heads)
            atomic_npz("CV_PREDICTIONS.npz",head_index=np.array(vc_head),support_union_index=np.array(vc_member),label=np.array(vc_y),prediction=np.array(vc_pred),scores=np.stack(vc_scores))
            seal={"status":"ALL_360_HEADS_AND_24_SELECTIONS_FROZEN_BEFORE_QUERY","utc":utc(),"choices":choices,"heads_sha256":sha(ROOT/"HEADS.npz"),"head_identities_sha256":sha(ROOT/"HEAD_IDENTITIES.json"),"cv_predictions_sha256":sha(ROOT/"CV_PREDICTIONS.npz"),"protocol_sha256":sha(ROOT/"PROTOCOL.json"),"execution_plan_sha256":sha(ROOT/"EXECUTION_PLAN.json"),"query_scores_computed":False}
            atomic_json("SELECTION_SEAL.json",seal)
            phase="query"
            ledger["query_start_utc"]=utc();ledger["selection_seal_sha256"]=sha(ROOT/"SELECTION_SEAL.json")
            query_features=extract(model,z,query,"query",protocol,ledger)[:,0,:]
            guard(protocol)
            t=time.perf_counter()
            allscores=np.stack([scores(query_features,W,b) for W,b in zip(Wlist,blist)])
            assert allscores.shape==(360,len(query),10) and np.isfinite(allscores).all()
            ledger["query_score_seconds"]=time.perf_counter()-t
            preds=allscores.argmax(-1);qy=np.array([r["label"] for r in query])
            atomic_json("QUERY_IDENTITIES.json",query)
            atomic_npz("QUERY_PREDICTIONS.npz",scores=allscores,prediction=preds,label=qy,head_index=np.arange(360),query_index=np.arange(len(query)))
            ledger["query_scores_computed"]=True;ledger["query_prediction_count"]=int(preds.size)
            ledger["final_state_hash"]=state_hash(model)
            assert ledger["initial_state_hash"]==ledger["final_state_hash"]
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            for name,h in code.items(): assert sha(ROOT/name)==h,"execution source changed"
            ledger.update(status="COMPLETE",completed_utc=utc(),wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,query_images=len(query),support_originals=len(m["support_union"]),query_predictions_sha256=sha(ROOT/"QUERY_PREDICTIONS.npz"),all_encoder_parameters_frozen=True)
            atomic_json("RUN_RECEIPT.json",ledger)
            atomic_json("STATUS.json",{"status":"COMPLETE","head_fits":360,"query_images":len(query),"query_predictions":int(preds.size),"utc":utc()})
            print(json.dumps({"phase":"complete","fits":360,"query_images":len(query),"wall_seconds":ledger["wall_seconds"],"peak_rss_bytes":ledger["peak_rss_bytes"]}),flush=True)
    except BaseException as exc:
        ledger.update(status="INCOMPLETE",phase=phase,error=repr(exc),ended_utc=utc(),wall_seconds=time.perf_counter()-start)
        atomic_json("RUN_RECEIPT.json",ledger);atomic_json("STATUS.json",ledger)
        raise

if __name__=="__main__":main()

