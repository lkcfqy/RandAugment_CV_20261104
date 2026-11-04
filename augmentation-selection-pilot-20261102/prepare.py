"""Identity/split freeze before model utility; local assets only."""
from common import *
import zipfile,collections,datetime
assert not (ROOT/"PROTOCOL.json").exists()
start=datetime.datetime(2026,11,2,4,20,tzinfo=datetime.timezone.utc)
ziphash=sha(ZIP);assert ziphash=="b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
weighthash=sha(MODEL/"model.safetensors");assert weighthash=="ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
oldpath=PROJECT/"research-frofa-pilot-20261030/manifest.json"
oldhash=sha(oldpath);assert oldhash=="0b4700a0311be08335ce2a55959dfea1a707c0493bc7fdf788e4b5df0dc48d73"
old=json.loads(oldpath.read_text());classes=old["classes"];assert len(classes)==10
audit={"zip_sha256":ziphash,"zip_bytes":ZIP.stat().st_size,"original_manifest_sha256":oldhash,"class_counts":{},"decoded_images":0}
t=time.perf_counter()
seen={};inventory={}
with zipfile.ZipFile(ZIP) as z:
    names=[n for n in z.namelist() if n.endswith(".jpg")]
    assert len(names)==len(set(names))==27000
    for n in sorted(names):
        assert not Path(n).is_absolute() and ".." not in Path(n).parts
        cls=Path(n).parts[-2];assert cls in classes
        im=Image.open(io.BytesIO(z.read(n))).convert("RGB");assert im.size==(64,64)
        h=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
        if h in seen: assert seen[h][0]==cls,"cross-label pixel duplicate"
        else: seen[h]=(cls,n)
        inventory[n]=h;audit["decoded_images"]+=1
        audit["class_counts"][cls]=audit["class_counts"].get(cls,0)+1
for row in old["rows"]: assert inventory[row["member"]]==row["rgb_sha256"]
audit["unique_rgb"]=len(seen);audit["decoded_seconds"]=time.perf_counter()-t
pool={c:[r for r in old["rows"] if r["class"]==c and r["split"]=="trainpool"] for c in classes}
tests={c:[r for r in old["rows"] if r["class"]==c and r["split"]=="test"] for c in classes}
assert all(len(pool[c])==100 and len(tests[c])==300 for c in classes)
query=[]
for c in classes:
    for r in sorted(tests[c],key=lambda r:(key("query",r["rgb_sha256"]),r["member"]))[:100]:
        query.append({k:r[k] for k in ["member","class","label","rgb_sha256","split"]})
episodes=[];selected={}
for e in range(12):
    full=[]
    for c in classes:
        rank=sorted(pool[c],key=lambda r:(key("support",e,r["rgb_sha256"]),r["member"]))[:16]
        for j,r in enumerate(rank):
            item={k:r[k] for k in ["member","class","label","rgb_sha256","split"]}
            item.update(rank_in_class=j,fold=j%4)
            full.append(item);selected[r["member"]]={k:r[k] for k in ["member","class","label","rgb_sha256","split"]}
    for B in [8,16]:
        rows=[x for x in full if x["rank_in_class"]<B]
        assert len(rows)==B*10
        assert collections.Counter((x["label"],x["fold"]) for x in rows)==collections.Counter({(c,f):B//4 for c in range(10) for f in range(4)})
        episodes.append({"episode":e,"B":B,"support":rows})
support=sorted(selected.values(),key=lambda r:r["member"])
sids={r["rgb_sha256"] for r in support};qids={r["rgb_sha256"] for r in query}
assert len(sids)==len(support) and len(qids)==len(query) and not sids&qids
overlap=[]
for a in range(12):
    A={r["member"] for r in episodes[a*2+1]["support"]}
    for b in range(a):
        B={r["member"] for r in episodes[b*2+1]["support"]}
        overlap.append({"a":a,"b":b,"shared_originals_B16":len(A&B)})
manifest={"classes":classes,"salt":SALT,"support_union":support,"query":query,"episodes":episodes,"historical_split_reuse":True,"query_images_per_class":100,"query_fresh_confirmation":False,"episode_pair_overlap":overlap}
atomic_json("IDENTITIES.json",manifest)
protocol={"status":"FROZEN_BEFORE_MODEL_UTILITY","frozen_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
"task_start_utc":start.isoformat(),"deadline_epoch":start.timestamp()+7200,"max_rss_bytes":12*1024**3,"max_artifact_bytes":500*1024**2,
"salt":SALT,"encoder":"facebook/dinov2-small","encoder_revision":"ed25f3a31f01632728cabb09d1542f84ab7b0056",
"encoder_weight_sha256":weighthash,"encoder_config_sha256":sha(MODEL/"config.json"),"encoder_processor_sha256":sha(MODEL/"preprocessor_config.json"),
"zip_sha256":ziphash,"previous_manifest_sha256":oldhash,"identities_sha256":sha(ROOT/"IDENTITIES.json"),
"cpu_threads":4,"interop_threads":1,"device":"cpu","dtype_encoder":"float32","dtype_ridge_scores":"float64","feature":"final layernorm CLS384, per-view unit L2",
"policies":POLICIES,"clean":"PIL RGB; bicubic resize shorter256; center crop224",
"weak":{"RRC_size":224,"scale":[.5,1.],"ratio":[.75,4/3],"interpolation":"bicubic","horizontal_flip_probability":.5},
"ra":{"after":"same weak geometry per identity/view","num_ops":2,"magnitude":9,"bins":31,"interpolation":"bilinear","fill_RGB":[124,116,104]},
"normalization":{"mean":[.485,.456,.406],"std":[.229,.224,.225]},
"views_per_original":4,"clean_view_compression":"four identical clean views represented by one encoder feature then exactly replicated for ridge",
"augmentation_seed":"SHA256(salt|geometry or randaugment|decoded RGB SHA256|view0..3), first64bits masked to63; CPU Torch fork_rng",
"ridge":{"lambda":LAMBDA,"targets":"onehot10","intercept":"unpenalized train-only weighted centering","objective":"mean_original mean_view squared residual + lambda Frobenius squared W","original_total_weight":"1/N irrespective of view count"},
"CV":{"folds":4,"fold_assignment":"per-class rank mod4, consistent B8 subset B16","validation_input":"clean original once","selection":"pooled correct count over all original out-of-fold predictions","tie_order":POLICIES},
"expected_fits":360,"expected_selections":24,"query_default_per_class":100,"query_reduction_before_any_utility_only":[100,50,25],
"query_gate":"all 360 heads and 24 CV choices sealed before query extraction/evaluation",
"primary_metrics":["selected minus weak full-fit query accuracy","selected minus RA full-fit query accuracy"],
"mandatory":"all fixed strategies, both budgets, every episode; harmful frequency separately vs weak/RA; posthoc best regret only diagnostic",
"diagnostic":"CV ranking vs mean four fold-model query accuracy vs full-fit query ranking, same tie order; not pure causal decomposition",
"limits":["exploratory reused data","correlated query land patches and overlapping supports; no independent-scene or independence claim","fixed ridge and 4 Monte Carlo views only","no thresholds demanding 1 or2pp; no effect-based stopping","not CLAP/FroFA reproduction or novel theory"],
"new_download_install_upload_CU":0}
atomic_json("PROTOCOL.json",protocol)
atomic_json("ASSET_AUDIT.json",audit)
print(json.dumps({"status":protocol["status"],"all_images":audit["decoded_images"],"classes":len(classes),"support_union":len(support),"query":len(query),"expected_fits":360,"protocol_sha256":sha(ROOT/"PROTOCOL.json"),"elapsed":audit["decoded_seconds"]}))

