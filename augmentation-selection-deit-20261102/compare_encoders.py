"""Related-encoder comparison; no model runs and no new selection rule."""
from common import *
import csv,collections
OLD=PROJECT/"augmentation-selection-pilot-20261102"
assert sha(ROOT/"IDENTITIES.json")==sha(OLD/"IDENTITIES.json")
assert json.loads((ROOT/"QUERY_IDENTITIES.json").read_text())==json.loads((OLD/"QUERY_IDENTITIES.json").read_text())
sources={"dinov2_small":OLD,"deit_tiny":ROOT}
summaries={name:json.loads((path/"SUMMARY.json").read_text()) for name,path in sources.items()}
rows=[];extra={}
for name,s in summaries.items():
    assert s["status"]=="COMPLETE_RECOMPUTED" and s["fits"]==360 and s["query_images"]==1000
    for x in s["all_settings"]: rows.append({"encoder":name,**x})
    extra[name]={}
    for B in [8,16]:
        r=[x for x in s["all_settings"] if x["B"]==B]
        full=[[x["full_clean"],x["full_weak"],x["full_ra"]] for x in r]
        fold=[[x["fold_query_clean"],x["fold_query_weak"],x["fold_query_ra"]] for x in r]
        extra[name][str(B)]={"full_query_tied_best_count":sum(sum(v==max(vals) for v in vals)>1 for vals in full),
          "fold_query_tied_best_count":sum(sum(v==max(vals) for v in vals)>1 for vals in fold),
          "fixed_weak_is_query_optimal_count":sum(x["full_weak"]==max(x["full_clean"],x["full_weak"],x["full_ra"]) for x in r),
          "oracle_minus_fixed_weak_mean_pp":float(np.mean([max(x["full_clean"],x["full_weak"],x["full_ra"])-x["full_weak"] for x in r])),
          "query_best_policy_counts_with_fixed_tie_rule":dict(collections.Counter(x["full_query_best"] for x in r))}
joint=[]
for B in [8,16]:
    for e in range(12):
        a=next(x for x in summaries["dinov2_small"]["all_settings"] if x["B"]==B and x["episode"]==e)
        b=next(x for x in summaries["deit_tiny"]["all_settings"] if x["B"]==B and x["episode"]==e)
        joint.append({"episode":e,"B":B,"dinov2_choice":a["selected"],"deit_choice":b["selected"],"same_choice":a["selected"]==b["selected"],
          "dinov2_selected_minus_weak_pp":a["selected_minus_weak_pp"],"deit_selected_minus_weak_pp":b["selected_minus_weak_pp"],
          "dinov2_selected_minus_ra_pp":a["selected_minus_ra_pp"],"deit_selected_minus_ra_pp":b["selected_minus_ra_pp"],
          "dinov2_regret_pp":a["oracle_regret_pp"],"deit_regret_pp":b["oracle_regret_pp"]})
for fn,table in [("BOTH_ENCODERS_EPISODES.csv",rows),("PAIRED_ENCODER_COMPARISON.csv",joint)]:
    with (ROOT/fn).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
result={"status":"COMPLETE_BOTH_ENCODERS","related_control_not_new_confirmation":True,"models":["dinov2_small","deit_tiny"],
"source_summary_sha256":{n:sha(p/"SUMMARY.json") for n,p in sources.items()},
"identities_same":True,"query_order_same":True,"weak_ra_input_tensor_parity":json.loads((ROOT/"SECOND_ENCODER_CONTRACT_TEST.json").read_text()),
"clean_preprocessing_difference":"DINOv2 resize256, DeiT official timm resize248; both center224 bicubic with identical normalization",
"same_tie_rule":"highest pooled CV integer correct; ties clean then weak then ra; diagnostic ranking uses same policy order, strict loss separately",
"summary_by_encoder":{n:s["summaries"] for n,s in summaries.items()},"tie_and_headroom":extra,"all_48_settings":rows,"paired_24_settings":joint,
"same_selected_policy_count_by_B":{str(B):sum(r["same_choice"] for r in joint if r["B"]==B) for B in [8,16]},
"interpretation_limits":["same reused exploratory query set and overlapping supports","not pure-weight causal attribution; architecture/pretraining/native clean input differ","posthoc query-best not deployable","no third run launched"]}
atomic_json("BOTH_ENCODERS_SUMMARY.json",result)
md=["# All settings from both encoders","","All percentages / percentage-point differences are descriptive. Every prespecified setting is included. Same query identities; native clean preprocessing differs.",""]
for name in sources:
    for B in [8,16]:
        md += [f"## {name}, B={B}","","| Episode | Clean | Weak | RA | CV choice | Selected-W | Selected-RA | Oracle regret |","|---:|---:|---:|---:|:---:|---:|---:|---:|"]
        for x in rows:
            if x["encoder"]==name and x["B"]==B:
                md.append(f'| {x["episode"]} | {x["full_clean"]:.1f} | {x["full_weak"]:.1f} | {x["full_ra"]:.1f} | {x["selected"]} | {x["selected_minus_weak_pp"]:+.1f} | {x["selected_minus_ra_pp"]:+.1f} | {x["oracle_regret_pp"]:.1f} |')
        md+=[""]
(ROOT/"ALL_EPISODES_BOTH.md").write_text("\n".join(md)+"\n")
print(json.dumps({"ties_headroom":extra,"same_choices":result["same_selected_policy_count_by_B"]},indent=2))

