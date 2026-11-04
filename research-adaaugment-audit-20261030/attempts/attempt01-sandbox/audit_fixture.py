"""Independent CPU-only audit. Downloaded source is parsed/hashed, never imported.

See SOURCE_MAP.md for every source-to-fixture correspondence and simplification.
The executable portions use the already installed PyTorch library only.
"""
from __future__ import annotations

import ast
from collections import Counter
import csv
import datetime
import hashlib
import inspect
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
CHECKS: list[dict] = []


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def check(name, passed, details=None):
    CHECKS.append({"name": name, "passed": bool(passed), "details": details})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantize(m):
    # AdaAugment.py:637; organize_transform.py:8 -> PARAMETER_MAX=29.
    return min(int(29 * float(m)) + 1, 29)


def initial_value(sid):
    # Synthetic stand-in for a completed warmup cache (train.py:120-131).
    return float(torch.tensor(0.05 + sid * 0.01, dtype=torch.float32).item())


def proposal_value(sid, visit):
    return float(torch.tensor((0.55 if visit == 1 else 0.75) + sid * 0.01,
                              dtype=torch.float32).item())


def value_version(sid, value):
    for version, candidate in [(0, initial_value(sid)), (1, proposal_value(sid, 1)),
                               (2, proposal_value(sid, 2))]:
        if value == candidate:
            return version
    return -1


class SyntheticCacheDataset(Dataset):
    """Tensor cache is faithful in type/update semantics; image and losses are toy."""

    def __init__(self, sample_count=8, event_queue=None):
        self.MAGNITUDE = torch.zeros(sample_count, dtype=torch.float32, device="cpu")
        for sid in range(sample_count):
            self.MAGNITUDE[sid] = initial_value(sid)
        self.event_queue = event_queue
        self.is_magnitude = True

    def __len__(self):
        return len(self.MAGNITUDE)

    def set_magnitude(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude  # Dataset.py:72-73, 117-118.

    def __getitem__(self, index):
        # Source reads .item(): the materialized output has no live cache alias.
        cached = self.MAGNITUDE[index].item()
        level = quantize(cached)
        worker = get_worker_info()
        record = {"sample_id": index, "cached_action": cached, "applied_level": level,
                  "cached_version": value_version(index, cached),
                  "cache_shared_in_worker": self.MAGNITUDE.is_shared(),
                  "worker_id": -1 if worker is None else worker.id,
                  "worker_pid": os.getpid(), "materialized_monotonic_ns": time.monotonic_ns(),
                  "materialized_input": [index / 10, level / 29],
                  "is_magnitude_in_worker": self.is_magnitude}
        if self.event_queue is not None:
            self.event_queue.put({"sid": index, "cached_version": record["cached_version"],
                                  "worker_pid": os.getpid()})
        return record


def identity_collate(batch):
    return batch


def toy_reward(materialized_input, balance=0.25):
    # No neural network. Retain source reward algebra at a noncancelling lambda.
    sample_term, level_fraction = materialized_input
    none_loss = 0.2 + sample_term
    ada_loss = none_loss + level_fraction
    full_loss = none_loss + 1.2
    return (1 - balance) * (ada_loss - none_loss) + balance * (full_loss - ada_loss)


def credit_record(ds, item, visit, mode):
    sid = item["sample_id"]
    before = tuple(item["materialized_input"])
    reward = toy_reward(before)  # Materialized before proposing a new action.
    proposed = proposal_value(sid, visit)
    ds.set_magnitude(sid, proposed)
    cached_after_proposal = ds.MAGNITUDE[sid].item()
    # Paired intervention: two new actions on exactly the same fixed input.
    alternatives = []
    for alternative in (0.2, 0.8):
        ds.set_magnitude(sid, alternative)
        alternatives.append(toy_reward(before))
    ds.set_magnitude(sid, proposed)
    record = dict(item, visit=visit, mode=mode, proposed_action=proposed,
                  cached_action_after_proposal=cached_after_proposal,
                  reward=reward, reward_origin=f"sid={sid}/cached_version={item['cached_version']}/level={item['applied_level']}",
                  logprob_action=proposed, counterfactual_new_actions=[0.2, 0.8],
                  counterfactual_rewards=alternatives,
                  fixed_input_reward_invariant=reward == alternatives[0] == alternatives[1],
                  logprob_action_matches_reward_action=proposed == item["cached_action"])
    return record


def loader_case(case):
    start = time.monotonic()
    ds = SyntheticCacheDataset(PROTOCOL["synthetic_samples"])
    kwargs = dict(batch_size=128, shuffle=True, num_workers=case["workers"],
                  pin_memory=False, persistent_workers=case["persistent"],
                  collate_fn=identity_collate,
                  generator=torch.Generator(device="cpu").manual_seed(PROTOCOL["loader_seed"]))
    if case["workers"]:
        kwargs.update(multiprocessing_context=case["context"], prefetch_factor=2, timeout=45)
    loader = DataLoader(ds, **kwargs)
    records, worker_sets = [], []
    shared_before = ds.MAGNITUDE.is_shared()
    for visit in (1, 2):
        current_pids = set()
        for batch in loader:
            for item in batch:
                current_pids.add(item["worker_pid"])
                records.append(credit_record(ds, item, visit, case["name"]))
        worker_sets.append(sorted(current_pids))
    # Controlled cleanup of the installed library's iterator, no repo execution.
    if loader._iterator is not None:
        loader._iterator._shutdown_workers()
    visit2 = [r for r in records if r["visit"] == 2]
    expected_v2 = 0 if case["persistent"] and case["context"] == "fork" else 1
    check(case["name"] + "/16_records", len(records) == 16)
    check(case["name"] + "/fixed_input_reward_invariant",
          all(r["fixed_input_reward_invariant"] for r in records))
    check(case["name"] + "/new_logprob_differs_from_reward_action",
          all(not r["logprob_action_matches_reward_action"] for r in records))
    check(case["name"] + "/next_visit_cache", all(r["cached_version"] == expected_v2 for r in visit2),
          {"expected": expected_v2, "observed": [r["cached_version"] for r in visit2]})
    summary = dict(case, records=len(records), wall_seconds=time.monotonic()-start,
                   cache_shared_before=shared_before, cache_shared_after=ds.MAGNITUDE.is_shared(),
                   worker_pids_by_visit=worker_sets,
                   next_visit_cache_versions=[r["cached_version"] for r in visit2],
                   actual_prefetch_factor=loader.prefetch_factor,
                   batch_size=loader.batch_size, pin_memory=loader.pin_memory)
    return records, summary


def prefetch_case(factor):
    """Force a documented queue boundary; no timing guess about when reads happened.

    8 distinct IDs, two visits within one iterator. Waiting for explicit read
    acknowledgements occurs before any cache write. This is a stress fixture,
    not the official no-replacement epoch sampler.
    """
    context = mp.get_context("spawn")
    events = context.Queue()
    ds = SyntheticCacheDataset(8, events)
    loader = DataLoader(ds, batch_size=1, sampler=list(range(8))*2, num_workers=8,
                        multiprocessing_context=context, prefetch_factor=factor,
                        persistent_workers=False, collate_fn=identity_collate,
                        pin_memory=False, timeout=45)
    iterator = iter(loader)
    expected_initial = min(16, 8*factor)
    acknowledgements = [events.get(timeout=20) for _ in range(expected_initial)]
    for sid in range(8):
        ds.set_magnitude(sid, proposal_value(sid, 1))
    records = []
    visits = Counter()
    for batch in iterator:
        for item in batch:
            visits[item["sample_id"]] += 1
            records.append(dict(item, visit=visits[item["sample_id"]]))
    while len(acknowledgements) < 16:
        acknowledgements.append(events.get(timeout=20))
    events.close()
    events.join_thread()
    expected = [0]*8 + ([1]*8 if factor == 1 else [0]*8)
    observed = [r["cached_version"] for r in records]
    check(f"prefetch{factor}/read_versions", observed == expected,
          {"expected":expected,"observed":observed})
    return {"factor": factor, "initial_acknowledged_reads": expected_initial,
            "records": records, "read_acknowledgements": acknowledgements,
            "cache_shared": ds.MAGNITUDE.is_shared()}


def score_gradient_test():
    cfg = PROTOCOL["beta"]
    torch.random.default_generator.manual_seed(cfg["seed"])
    theta = torch.tensor([cfg["alpha"], cfg["beta"]], dtype=torch.float64,
                         device="cpu", requires_grad=True)
    dist = torch.distributions.Beta(theta[0], theta[1])
    action = dist.sample((cfg["samples"],)).detach()
    old_action = dist.sample((cfg["samples"],)).detach()
    score = torch.stack((torch.log(action)-torch.digamma(theta[0])+torch.digamma(theta.sum()),
                         torch.log1p(-action)-torch.digamma(theta[1])+torch.digamma(theta.sum())), dim=1).detach()
    results = {}
    for name, reward, expected in [("same_action", action, cfg["same_action_expected"]),
                                   ("independent_old", old_action, cfg["independent_old_expected"])]:
        raw_terms = reward[:, None]*score
        mean = raw_terms.mean(0)
        se = raw_terms.std(0, unbiased=True)/math.sqrt(cfg["samples"])
        z = 2.5758293035489004
        lo, hi = mean-z*se, mean+z*se
        # Raw ascent score. Official minimized actor loss is negative and sums.
        gradient = torch.autograd.grad((dist.log_prob(action)*reward).mean(), theta,
                                       retain_graph=True)[0]
        target = torch.tensor(expected, dtype=torch.float64, device="cpu")
        check(name+"/autograd_matches_raw_score", (gradient-mean).abs().max().item() <= cfg["autograd_vs_score_tolerance"])
        check(name+"/analytic_expectation_in_99ci", bool(((lo <= target)&(target <= hi)).all()))
        check(name+"/predeclared_abs_tolerance", bool(((mean-target).abs() <= cfg["absolute_tolerance"]).all()))
        results[name] = {"mean_raw_score_gradient":mean.tolist(),"autograd_mean_gradient":gradient.tolist(),
                         "standard_errors":se.tolist(),"ci99":[[float(lo[j]),float(hi[j])] for j in range(2)],
                         "analytic_expectation":expected,"official_actor_loss_sum_gradient":(-cfg["samples"]*gradient).tolist()}
    check("same_action/direction", results["same_action"]["mean_raw_score_gradient"][0]>0 and
          results["same_action"]["mean_raw_score_gradient"][1]<0)
    check("beta/actor_unchanged", theta.tolist() == [2.0,3.0] and theta.grad is None)
    # Check softplus+1 parameterization without building the official network.
    logits = torch.tensor([math.log(math.expm1(1.0)),math.log(math.expm1(2.0))],
                          dtype=torch.float64, device="cpu", requires_grad=True)
    ab = torch.nn.functional.softplus(logits)+1
    g_logits = torch.autograd.grad((torch.distributions.Beta(ab[0],ab[1]).log_prob(action)*action).mean(),logits)[0]
    chain = torch.tensor(results["same_action"]["autograd_mean_gradient"],dtype=torch.float64)*torch.sigmoid(logits.detach())
    check("beta/softplus_chain_rule", (g_logits-chain).abs().max().item() < 1e-12)
    results.update(samples=cfg["samples"],seed=cfg["seed"],zero_critic=True,
                   tensor_devices=[str(action.device),str(old_action.device),str(theta.device)],
                   sample_sha256={"new":hashlib.sha256(action.numpy().tobytes()).hexdigest(),
                                  "old":hashlib.sha256(old_action.numpy().tobytes()).hexdigest()},
                   softplus_logit_gradient=g_logits.tolist(),
                   note="Mean raw score only; no optimizer, Adam inference, trajectory claim or accuracy estimate")
    return results


def action_bundle(magnitude, operations=("Rotate",)):
    """Complete realized synthetic action bundle. Random directions fixed positive.

    Operations with ignored levels canonicalize to no effective parameter.
    Tokens for crop/flip/cutout are identical controlled constants, included in
    the full record so omitting them cannot silently pass the multiset test.
    """
    level = quantize(magnitude)
    actual = []
    for order, op in enumerate(operations):
        if op == "Rotate":
            parameter = int(level*30/29)
        elif op == "TranslateX":
            parameter = int(level*10/29)
        elif op == "Posterize":
            parameter = 4-int(level*4/29)
        elif op == "Brightness":
            parameter = float(level)*1.8/29+0.1
        elif op in ("Identity", "AutoContrast", "Equalize"):
            parameter = None
        else:
            raise ValueError(op)
        actual.append({"op":op,"level":level,"physical_parameter":parameter,
                       "direction":1,"apply":True,"order":order})
    return {"raw_magnitude":magnitude,"operations":actual,"shared_level":level,
            "other_randomness":{"crop_box":[0,0,2,2],"flip":False,"cutout_box":[0,0,0,0]}}


def full_key(bundle):
    return json.dumps(bundle,sort_keys=True,separators=(",",":"))


def effective_key(bundle):
    return json.dumps({"operations":[{k:v for k,v in op.items() if k != "level"}
                                     for op in bundle["operations"]],
                       "other_randomness":bundle["other_randomness"]},sort_keys=True)


def permutation_test():
    scenarios = [
        ("singleton",[action_bundle(0.2)]),
        ("all_same",[action_bundle(0.2) for _ in range(4)]),
        ("same_after_quantization",[action_bundle(m) for m in (0.201,0.202,0.203,0.204)]),
        ("same_physical_parameter",[action_bundle(m,("Posterize",)) for m in (0.04,0.08,0.12,0.16)]),
        ("magnitude_ignored",[action_bundle(m,("Identity",)) for m in (0.1,0.3,0.6,0.9)]),
        ("shared_multi_operation",[action_bundle(m,("Rotate","Brightness")) for m in (0.1,0.3,0.6,0.9)]),
        ("known_utility",[action_bundle(m) for m in (0.1,0.3,0.6,0.9)])
    ]
    rng = random.Random(PROTOCOL["permutation"]["seed"])
    out = []
    for name,bundles in scenarios:
        n=len(bundles)
        keys=[effective_key(b) for b in bundles]
        coverage=sum(any(keys[i]!=keys[j] for j in range(n) if i!=j) for i in range(n))/n
        p=list(range(n)); rng.shuffle(p)
        random_bundles=[bundles[j] for j in p]
        perms=list(itertools.permutations(range(n)))
        metrics=[]
        for perm in perms:
            moved=[bundles[j] for j in perm]
            preserved=Counter(map(full_key,moved))==Counter(map(full_key,bundles))
            joint=all(len({op["level"] for op in b["operations"]})==1 for b in moved)
            metrics.append({"permutation":list(perm),"actual_parameter_change_rate":sum(keys[i]!=keys[j] for i,j in enumerate(perm))/n,
                            "raw_magnitude_change_rate":sum(bundles[i]["raw_magnitude"]!=bundles[j]["raw_magnitude"] for i,j in enumerate(perm))/n,
                            "multiset_preserved":preserved,"joint_shared_level_preserved":joint})
        check(name+"/all_permutations_preserve_full_multiset", all(m["multiset_preserved"] for m in metrics))
        check(name+"/all_permutations_keep_shared_intensity", all(m["joint_shared_level_preserved"] for m in metrics))
        expected_coverage=1.0 if name in ("shared_multi_operation","known_utility") else 0.0
        check(name+"/effective_coverage",coverage==expected_coverage)
        random_change=sum(keys[i]!=keys[j] for i,j in enumerate(p))/n
        row={"name":name,"n":n,"full_original_action_multiset":bundles,
             "random_permutation":p,"random_full_action_multiset":random_bundles,
             "permutable_effective_coverage":coverage,"random_actual_parameter_change_rate":random_change,
             "random_raw_magnitude_change_rate":sum(bundles[i]["raw_magnitude"]!=bundles[j]["raw_magnitude"] for i,j in enumerate(p))/n,
             "exact_mean_actual_parameter_change_rate":sum(m["actual_parameter_change_rate"] for m in metrics)/len(metrics),
             "exact_probability_all_parameters_change":sum(m["actual_parameter_change_rate"]==1 for m in metrics)/len(metrics),
             "all_index_permutations":metrics}
        if name=="known_utility":
            # Deliberately known synthetic utility: receiver i likes donor i.
            utility=[[int(i==j) for j in range(n)] for i in range(n)]
            scores=[sum(utility[i][j] for i,j in enumerate(perm))/n for perm in perms]
            bad=[1,2,3,0]
            row["utility"]={"table":utility,"good_pairing":list(range(n)),"good":1.0,
                            "bad_pairing":bad,"bad":sum(utility[i][j] for i,j in enumerate(bad))/n,
                            "random":sum(utility[i][j] for i,j in enumerate(p))/n,
                            "all_permutations_mean":sum(scores)/len(scores)}
            check("utility/known_good_bad_random", row["utility"]["bad"]==0 and row["utility"]["all_permutations_mean"]==0.25)
            check("permutation/not_always_derangement", row["exact_probability_all_parameters_change"]==9/24)
        out.append(row)
    # Reject scalar-only independent shuffles that break a known joint bundle.
    good=[action_bundle(m,("Rotate","Brightness")) for m in (0.1,0.3,0.6,0.9)]
    broken=json.loads(json.dumps(good))
    for i in range(4):
        broken[i]["operations"][1]=good[(i+1)%4]["operations"][1]
    check("multi_operation/reject_broken_joint_multiset",Counter(map(full_key,broken))!=Counter(map(full_key,good)))
    check("multi_operation/reject_broken_shared_level",any(len({o["level"] for o in b["operations"]})>1 for b in broken))
    return out


def source_checks():
    receipts=json.loads((ROOT/"source-receipts.json").read_text())
    for rec in receipts:
        check("source_hash/"+rec["path"],sha256(ROOT/rec["path"])==rec["sha256"])
    python_sources=list((ROOT/"sources").rglob("*.py"))
    imports={}
    for path in python_sources:
        tree=ast.parse(path.read_text(),filename=str(path))
        imports[str(path.relative_to(ROOT/"sources"))]=[
            {"line":node.lineno,"text":ast.get_source_segment(path.read_text(),node)}
            for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
    # Text anchors plus all file hashes bind our mapping to inspected lines.
    anchors={"Dataset.py":{52:"torch.zeros(50000)",73:"self.MAGNITUDE[idx] = magnitude",84:"self.MAGNITUDE[index].item()",118:"self.MAGNITUDE[idx] = magnitude",123:"self.MAGNITUDE[index].item()"},
             "train.py":{72:"num_workers=8",131:"trainset.set_MAGNITUDE",155:"net(ada_aug_inputs",163:"loss(ada_outputs",167:"agent.action(none_state)",169:"trainset.set_MAGNITUDE",171:"ada_loss.detach()",181:"agent.update(none_state, magnitude, ada_state, reward)",231:"train_first_epoch",232:"is_magnitude = True"},
             "A2C.py":{68:"dist.sample()",75:"dist.log_prob(action)",82:"actor_loss = -(log_prob"},
             "augmentation/adaaugment.py":{366:"PARAMETER_MAX = num_strengths - 1",636:"k=1",637:"min(int(PARAMETER_MAX * M)+1, PARAMETER_MAX)"}}
    for name,lines in anchors.items():
        text=(ROOT/"sources"/name).read_text().splitlines()
        for lineno,needle in lines.items():
            check(f"source_anchor/{name}:{lineno}",needle in text[lineno-1],text[lineno-1])
    write_json(OUT/"imports-and-source-anchors.json",{"imports":imports,"anchors":anchors})
    return len(python_sources)


def main():
    start=time.monotonic()
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    write_json(OUT/"environment.json",{"python":sys.version,"torch":torch.__version__,
               "executable":sys.executable,"default_start_method":mp.get_start_method(),
               "dataloader_signature":str(inspect.signature(DataLoader)),
               "cpu_only":True,"installed_library_used":str(Path(torch.__file__).resolve())})
    fatal=None
    try:
        files=source_checks()
        print(f"Source text checked: {files} Python files; no repository imports",flush=True)
        traces,summaries=[],[]
        for case in PROTOCOL["loader_cases"]:
            records,summary=loader_case(case)
            traces+=records; summaries.append(summary)
            write_json(OUT/"loader-traces.json",traces)
            write_json(OUT/"loader-summary.json",summaries)
            print("Loader",case["name"],"next-cache",summary["next_visit_cache_versions"],flush=True)
        prefetch=[]
        for factor in PROTOCOL["prefetch_probe"]["factors"]:
            prefetch.append(prefetch_case(factor))
            write_json(OUT/"prefetch.json",prefetch)
            print("Prefetch",factor,"verified",flush=True)
        gradients=score_gradient_test()
        write_json(OUT/"score-gradient.json",gradients)
        print("Gradient",json.dumps({k:gradients[k]["mean_raw_score_gradient"] for k in ("same_action","independent_old")}),flush=True)
        permutations=permutation_test()
        write_json(OUT/"permutations.json",permutations)
        with (OUT/"permutation-summary.csv").open("w",newline="") as f:
            fields=["name","n","permutable_effective_coverage","random_raw_magnitude_change_rate","random_actual_parameter_change_rate","exact_mean_actual_parameter_change_rate","exact_probability_all_parameters_change"]
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for row in permutations: writer.writerow({k:row[k] for k in fields})
        check("resources/no_target_training",True,"No target model constructed; no optimizer constructed")
        check("resources/formal_compute_under_600s",time.monotonic()-start<600)
    except Exception:
        fatal=traceback.format_exc()
        print(fatal,flush=True)
    finally:
        summary={"started_utc":started_utc,
                 "wall_seconds":time.monotonic()-start,"checks":len(CHECKS),
                 "passed":sum(c["passed"] for c in CHECKS),"failed":sum(not c["passed"] for c in CHECKS),
                 "fatal_error":fatal,"target_model_training_steps":0,"optimizer_steps":0,
                 "colab_CU":0,"accelerator_tensor_operations_requested":0,"real_data_downloads":0,"weight_downloads":0}
        write_json(OUT/"checks.json",CHECKS)
        write_json(OUT/"summary.json",summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 1 if fatal or any(not c["passed"] for c in CHECKS) else 0


if __name__=="__main__":
    raise SystemExit(main())
