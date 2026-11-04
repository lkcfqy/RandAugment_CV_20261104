"""Recompute every result from the sealed choices and complete persisted predictions."""
from common import *
import csv,collections

def signs(x):
    return [int(np.sign(x[a]-x[b])) for a,b in [(0,1),(0,2),(1,2)]]

def analyze():
    run=json.loads((ROOT/"RUN_RECEIPT.json").read_text());assert run["status"]=="COMPLETE"
    seal=json.loads((ROOT/"SELECTION_SEAL.json").read_text())
    heads=json.loads((ROOT/"HEAD_IDENTITIES.json").read_text())
    assert len(heads)==360 and len(seal["choices"])==24
    assert sha(ROOT/"HEADS.npz")==seal["heads_sha256"]
    assert sha(ROOT/"HEAD_IDENTITIES.json")==seal["head_identities_sha256"]
    assert sha(ROOT/"QUERY_PREDICTIONS.npz")==run["query_predictions_sha256"]
    assert run["query_start_utc"]>seal["utc"]
    with np.load(ROOT/"QUERY_PREDICTIONS.npz",allow_pickle=False) as q:
        scores_saved=q["scores"];pred=q["prediction"];y=q["label"]
        assert np.array_equal(pred,scores_saved.argmax(-1))
        assert np.array_equal(q["head_index"],np.arange(360))
        assert pred.shape==(360,len(y))
    with np.load(ROOT/"CV_PREDICTIONS.npz",allow_pickle=False) as cv:
        cvhead=cv["head_index"];cvy=cv["label"];cvpred=cv["prediction"]
        assert np.array_equal(cvpred,cv["scores"].argmax(-1))
    assert len({(h["episode"],h["B"],h["policy"],h["fold"]) for h in heads})==360
    correct=(pred==y[None,:]).sum(1)
    metrics=[]
    for h,n in zip(heads,correct):metrics.append({**h,"query_correct":int(n),"query_total":len(y),"query_accuracy_percent":float(100*n/len(y))})
    with (ROOT/"ALL_HEAD_METRICS.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    rows=[]
    for choice in seal["choices"]:
        e,B=choice["episode"],choice["B"];full=[];fold=[];computed_cv=[]
        for p in POLICIES:
            fulli=[h["index"] for h in heads if h["episode"]==e and h["B"]==B and h["policy"]==p and h["fold"]==-1]
            foldi=[h["index"] for h in heads if h["episode"]==e and h["B"]==B and h["policy"]==p and h["fold"]>=0]
            assert len(fulli)==1 and len(foldi)==4
            full.append(float(100*correct[fulli[0]]/len(y)))
            fold.append(float(100*correct[foldi].sum()/(4*len(y))))
            cvmask=np.isin(cvhead,foldi);assert int(cvmask.sum())==B*10
            computed_cv.append(int((cvpred[cvmask]==cvy[cvmask]).sum()))
        assert computed_cv==choice["cv_correct"] and choose(computed_cv)==choice["selected_index"]
        s=choice["selected_index"];f=choose(fold);o=choose(full)
        cvacc=[100*x/(B*10) for x in computed_cv]
        row={"episode":e,"B":B,"cv_clean":cvacc[0],"cv_weak":cvacc[1],"cv_ra":cvacc[2],"selected":POLICIES[s],
             "full_clean":full[0],"full_weak":full[1],"full_ra":full[2],"selected_accuracy":full[s],
             "selected_minus_weak_pp":full[s]-full[1],"selected_minus_ra_pp":full[s]-full[2],
             "selected_minus_clean_pp":full[s]-full[0],"oracle_regret_pp":max(full)-full[s],
             "fixed_policy_spread_pp":max(full)-min(full),
             "fold_query_clean":fold[0],"fold_query_weak":fold[1],"fold_query_ra":fold[2],
             "fold_query_best":POLICIES[f],"full_query_best":POLICIES[o],
             "cv_vs_fold_winner_diff":s!=f,"fold_vs_full_winner_diff":f!=o,
             "cv_choice_strictly_worse_on_fold_query":fold[s]<max(fold),
             "fold_choice_strictly_worse_on_full_query":full[f]<max(full),
             "cv_choice_is_full_query_optimal":full[s]==max(full)}
        cvsign,foldsign,fullsign=signs(cvacc),signs(fold),signs(full)
        row.update(cv_vs_fold_strict_pair_reversals=sum(a*b<0 for a,b in zip(cvsign,foldsign)),
                   fold_vs_full_strict_pair_reversals=sum(a*b<0 for a,b in zip(foldsign,fullsign)),
                   cv_vs_fold_pair_tie_changes=sum((a==0)!=(b==0) for a,b in zip(cvsign,foldsign)),
                   fold_vs_full_pair_tie_changes=sum((a==0)!=(b==0) for a,b in zip(foldsign,fullsign)))
        rows.append(row)
    with (ROOT/"EPISODE_RESULTS.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for B in [8,16]:
        r=[x for x in rows if x["B"]==B]
        desc={}
        for name in ["full_clean","full_weak","full_ra","selected_accuracy","selected_minus_weak_pp","selected_minus_ra_pp","selected_minus_clean_pp","oracle_regret_pp","fixed_policy_spread_pp"]:
            a=np.array([x[name] for x in r]);desc[name]={"mean":float(a.mean()),"median":float(np.median(a)),"sd_descriptive":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"all_12":a.tolist()}
        summaries[str(B)]={"descriptive":desc,"selected_counts":dict(collections.Counter(x["selected"] for x in r)),
            "selected_vs_weak_counts":{"better":sum(x["selected_minus_weak_pp"]>0 for x in r),"equal":sum(x["selected_minus_weak_pp"]==0 for x in r),"harmful":sum(x["selected_minus_weak_pp"]<0 for x in r)},
            "selected_vs_ra_counts":{"better":sum(x["selected_minus_ra_pp"]>0 for x in r),"equal":sum(x["selected_minus_ra_pp"]==0 for x in r),"harmful":sum(x["selected_minus_ra_pp"]<0 for x in r)},
            "selected_query_optimal_count":sum(x["cv_choice_is_full_query_optimal"] for x in r),
            "cv_selection_tied_count":sum(sum(v==max(x["cv_clean"],x["cv_weak"],x["cv_ra"]) for v in [x["cv_clean"],x["cv_weak"],x["cv_ra"]])>1 for x in r),
            "diagnostic_counts":{k:sum(x[k] for x in r) for k in ["cv_vs_fold_winner_diff","fold_vs_full_winner_diff","cv_choice_strictly_worse_on_fold_query","fold_choice_strictly_worse_on_full_query","cv_vs_fold_strict_pair_reversals","fold_vs_full_strict_pair_reversals","cv_vs_fold_pair_tie_changes","fold_vs_full_pair_tie_changes"]},
            "winner_change_categories":dict(collections.Counter(("CV_vs_fold:" + str(x["cv_vs_fold_winner_diff"]) + ",fold_vs_full:" + str(x["fold_vs_full_winner_diff"])) for x in r))}
    result={"status":"COMPLETE_RECOMPUTED","fits":360,"episodes":12,"budgets":[8,16],"query_images":len(y),"query_predictions":int(pred.size),"query_scores_sha256":sha(ROOT/"QUERY_PREDICTIONS.npz"),"selection_seal_sha256":sha(ROOT/"SELECTION_SEAL.json"),"summaries":summaries,"all_settings":rows,"caveats":["same fixed exploratory query set; overlapping supports","diagnostic oracle is not deployed","ranking comparisons are not pure causal decomposition","no pp success threshold or significance claim"]}
    atomic_json("SUMMARY.json",result)
    atomic_json("PREDICTIONS_SCHEMA.json",{"QUERY_PREDICTIONS.npz":{"scores":"float64 [360 heads, query images,10 classes], ridge scores not calibrated probabilities","prediction":"argmax class int","label":"true class per query","head_index":"row maps HEAD_IDENTITIES.json","query_index":"column maps QUERY_IDENTITIES.json"},"CV_PREDICTIONS.npz":{"head_index":"fold head row","support_union_index":"IDENTITIES.json support_union index","label":"original true class","prediction":"clean heldout-original prediction","scores":"float64 [heldout original records,10]"},"HEADS.npz":{"W":"[360,384,10] float64","b":"[360,10] float64"},"policy_order":POLICIES})
    print(json.dumps({"status":result["status"],"fits":360,"query_images":len(y),"summaries":{B:{"means":{k:v["mean"] for k,v in s["descriptive"].items()},"selected":s["selected_counts"],"vs_weak":s["selected_vs_weak_counts"],"vs_ra":s["selected_vs_ra_counts"],"diagnostic_counts":s["diagnostic_counts"]} for B,s in summaries.items()}},indent=2))

if __name__=="__main__":analyze()

