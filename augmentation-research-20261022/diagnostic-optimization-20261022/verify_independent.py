"""Independent CPU audit of SAVED eight-reset-step artifacts; never runs a model.

Usage: isolated-python verify_independent.py [archive-root] [--output FILE]
Default root and output are beside this script. Write output exclusively; do not
overwrite a prior audit. No main-module imports, image decoding, autograd, optimizer,
development logits, test data, GPU queries, or forward calls are used.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
PILOT = Path("/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/pilot-20261022")
EVENTS, INPUTS = [], {}
REPORT_ATOL, REPORT_RTOL = 1e-15, 2e-12
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    value = h.hexdigest()
    INPUTS[str(path)] = {"bytes": path.stat().st_size, "sha256": value}
    return value


def readj(path):
    sha(path)
    return json.loads(Path(path).read_text())


def check(name, ok, **detail):
    EVENTS.append({"check": name, "passed": bool(ok), **detail})
    return bool(ok)


def require(name, ok, **detail):
    if not check(name, ok, **detail):
        raise ValueError(name)


def close(name, actual, expected, atol=REPORT_ATOL, rtol=REPORT_RTOL):
    a, e = np.asarray(actual), np.asarray(expected)
    ok = a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all()
    delta = float(np.max(np.abs(a-e))) if ok and a.size else None
    return check(name, ok and np.allclose(a, e, atol=atol, rtol=rtol),
                 max_abs_difference=delta, atol=atol, rtol=rtol)


def compare(name, actual, expected):
    """Check expected fields, allowing explicitly harmless extra receipt fields."""
    if isinstance(expected, dict):
        require(name+".mapping", isinstance(actual, dict))
        for k, v in expected.items():
            require(name+"."+k+".present", k in actual)
            compare(name+"."+k, actual[k], v)
    elif isinstance(expected, list):
        require(name+".length", isinstance(actual, list) and len(actual) == len(expected))
        for i, v in enumerate(expected):
            compare(name+f"[{i}]", actual[i], v)
    elif isinstance(expected, float):
        close(name, actual, expected)
    else:
        check(name, actual == expected)


def state(path, key):
    sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(str(path)+".state", isinstance(payload, dict) and isinstance(payload.get(key), dict))
    arrays = {}
    for n, tensor in payload[key].items():
        require(str(path)+":"+n+".CPU_FP32_finite",
                isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()))
        arrays[n] = tensor.detach().contiguous().numpy()
    return payload, arrays


def state_hash(arrays, names=True):
    h = hashlib.sha256()
    for n, a in arrays.items():
        if names:
            h.update(n.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def features(path, keys):
    sha(path)
    with np.load(path, allow_pickle=False) as z:
        require(str(path)+".keys", set(z.files) == set(keys))
        values = [z[k].copy() for k in keys]
    for k, a in zip(keys, values):
        require(str(path)+":"+k+".shape_dtype_finite",
                a.shape == (2, 16, 768) and a.dtype == np.float32 and np.isfinite(a).all())
    return [a.astype(np.float64) for a in values]


def norm(a):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)))


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64)**2)))


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def audit(root):
    cfg, lock = readj(root/"protocol.json"), readj(root/"protocol-lock.json")
    require("protocol.hash", sha(root/"protocol.json") == lock["protocol_sha256"])
    for name, h in lock["code_sha256"].items():
        require("locked_code."+name, Path(name).name == name and sha(root/name) == h)
    require("locked_core_code_set", {"prepare_protocol.py", "diagnostic.py", "analyze.py"}.issubset(lock["code_sha256"]))
    require("original_common.hash", sha(PILOT/"common.py") == lock["original_common_sha256"])
    for field, rel in [("original_pilot_protocol_sha256", "protocol.json"),
                       ("fit_manifest_sha256", "fit-manifest.json"),
                       ("weight_sha256", "data/vit_b_16-c867db91.pth"),
                       ("protected_original_pilot_delivery_manifest_sha256", "delivery-manifest.json")]:
        require("protected_original."+rel, sha(PILOT/rel) == cfg[field])
    manifest, old_delivery = readj(PILOT/"fit-manifest.json"), readj(PILOT/"delivery-manifest.json")
    old_index = {r["relative_path"]: r for r in old_delivery["files"]}
    for rel in ("common.py", "protocol.json", "fit-manifest.json", "data/vit_b_16-c867db91.pth"):
        require("original_delivery."+rel, sha(PILOT/rel) == old_index[rel]["sha256"])
    check("fixed_model_and_resource_settings", cfg["model_seed"] == 20261022 and cfg["rank"] == 4
          and cfg["layouts"] == ["front", "back"] and cfg["blocks"] == {"front": [0,1,2], "back": [9,10,11]}
          and cfg["wall_cap_seconds"] == 600 and cfg["CPU_threads"] == 1
          and cfg["MPS_fallback"] is False and cfg["paced_sleep_multiple"] == 2)
    opt = cfg["optimizer"]
    compare("fixed_optimizer", opt, {"adapter_base_lr": .0003, "head_base_lr": .001,
            "adapter_actual_lr": .0003/111, "head_actual_lr": .001/111,
            "weight_decay": .01, "betas": [.9,.999], "eps": 1e-8,
            "total_original_updates": 1110, "logical_step": 0, "warmup_updates": 111})
    compare("fixed_oracle_tolerance", cfg["precision"]["manual_AdamW_parameter_oracle"],
            {"atol": 2e-10, "rtol": 2e-6})
    train = [r for r in manifest["rows"] if r["candidate_partition"] == "train"]
    require("retained_training_count", len(train) == 591 and len({r["source_id"] for r in train}) == 591)
    ordered = sorted(train, key=lambda r: hashlib.sha256(("opt-diagnostic-v1|20261022|"+r["source_id"]).encode()).hexdigest())
    fields = ["source_id", "class_id", "candidate_partition", "image_sha256", "decoded_RGB_sha256", "duplicate_cluster_id"]
    selected = []
    for i, r in enumerate(ordered[:96]):
        expected = {k: r[k] for k in fields}
        expected.update(role="gradient" if i < 64 else "probe", batch=i//16 if i < 64 else (i-64)//16)
        selected.append(expected)
    compare("fixed_hash_selection", cfg["selected_rows"], selected)
    require("96_disjoint_training_sources_and_clusters", len({r["source_id"] for r in selected}) == 96
            and len({r["duplicate_cluster_id"] for r in selected}) == 96
            and all(r["candidate_partition"] == "train" and 0 <= r["class_id"] < 37 for r in selected))
    inp = readj(root/"results/input-receipt.json")
    compare("input_scope", inp, {"training_only": True, "photos": 96, "gradient_photos": 64,
            "probe_photos": 32, "development_test_or_reserved_pixels_logits_or_accuracy_read": False})
    require("input_receipt_count", len(inp["rows"]) == 96)
    for i, (r, e) in enumerate(zip(inp["rows"], selected)):
        compare(f"input_receipt[{i}]", r, {k: e[k] for k in ("source_id", "role", "batch", "image_sha256")})
        check(f"input_receipt[{i}].tensor_digest_format", bool(HEX.fullmatch(r["tensor_sha256"])))
        require(f"input_receipt[{i}].compressed_training_photo_hash",
                sha(PILOT/"data/images"/(e["source_id"]+".jpg")) == e["image_sha256"])
    started, done = readj(root/"results/started.json"), readj(root/"results/completed.json")
    compare("started", started, {"protocol_sha256": lock["protocol_sha256"], "CPU_threads": 1, "MPS_fallback": False})
    compare("completed", done, {"status": "all8_reset_one_step_cases_completed", "protocol_sha256": lock["protocol_sha256"],
            "completed_reset_steps": 8, "sequential_steps_per_model": 1,
            "training_images_seen_for_gradients": 64, "training_only_probe_photos": 32,
            "development_accuracy_or_test_read": False, "wall_cap_seconds": 600})
    check("declared_UTC_order", stamp(cfg["at_utc"]) <= stamp(lock["at_utc"]) <= stamp(started["at_utc"]) < stamp(done["at_utc"]))
    check("600_second_cap", 0 < done["wall_seconds"] <= 600)
    check("no_structured_failure_marker", not (root/"diagnostic-failure.json").exists())
    init, fzero, rows, baselines, scales, pooled, headgrads = {}, {}, [], {}, {}, {}, {}
    all_resolved, norm_diagnostics = True, []
    for layout in ("front", "back"):
        folder = root/"results"/layout
        meta, a0 = state(folder/"initial-trainable.pt", "trainable_state")
        init[layout] = a0
        compare(layout+".initial_meta", meta, {"layout": layout, "protocol_sha256": lock["protocol_sha256"]})
        old_rel = "results/"+("front-weak" if layout == "front" else "back-weak")+"/initial-trainable-checkpoint.pt"
        require(layout+".old_initial_hash", sha(PILOT/old_rel) == old_index[old_rel]["sha256"])
        _, old = state(PILOT/old_rel, "trainable_state")
        expected_names = {"heads.head.weight", "heads.head.bias"}
        for block in cfg["blocks"][layout]:
            prefix = f"encoder.layers.encoder_layer_{block}.self_attention.parametrizations.in_proj_weight.0"
            expected_names.update(prefix+suffix for suffix in (".aq", ".bq", ".av", ".bv"))
        require(layout+".trainable_names", set(a0) == expected_names == set(old))
        check(layout+".parameter_count", sum(v.size for v in a0.values()) == 65317)
        for n in a0:
            check(layout+".original_initial."+n, np.array_equal(a0[n], old[n]))
            if n.endswith((".bq", ".bv")):
                check(layout+".B0."+n, np.count_nonzero(a0[n]) == 0 and a0[n].shape == (768,4))
            elif n.endswith((".aq", ".av")):
                check(layout+".A_shape."+n, a0[n].shape == (4,768))
        f0, rep = features(folder/"initial-features.npz", ["features", "repeat_features"])
        fzero[layout] = f0
        floors = [max(rms(rep[j]-f0[j]), np.finfo(np.float32).eps*rms(f0[j])) for j in range(2)]
        baselines[layout] = {"probe_RMS": [rms(f0[j]) for j in range(2)],
                            "repeat_noise_RMS": [rms(rep[j]-f0[j]) for j in range(2)], "floors": floors}
        scales[layout], pooled[layout] = [], []
        for batch in range(4):
            case = f"{layout}-batch{batch}"
            d = folder/f"batch{batch}"
            r = readj(d/"receipt.json")
            compare(case+".receipt", r, {"case": case, "layout": layout, "gradient_batch": batch,
                    "reset_from_initial_sha256": state_hash(a0), "initial_slot_sha256": state_hash(a0, False),
                    "optimizer_steps_in_this_fresh_state": 1, "A_data_gradients_all_exact0": True,
                    "B_data_gradients_all_nonzero": True, "manual_first_AdamW_parameter_oracle_passed": True,
                    "probe_feature_shape": [2,16,768]})
            gm, grads = state(d/"gradients.pt", "gradients")
            pm, after = state(d/"post-step-trainable.pt", "trainable_state")
            for name, meta in (("gradients", gm), ("after", pm)):
                compare(case+"."+name+".meta", meta, {"case": case, "protocol_sha256": lock["protocol_sha256"]})
            require(case+".matching_parameter_names", set(a0) == set(grads) == set(after) == set(r["parameter_records"]))
            if layout == "front":
                headgrads[batch] = {n: g.copy() for n,g in grads.items() if n.startswith("heads.")}
            else:
                for n, g in headgrads[batch].items():
                    close(case+".paired_head_gradient."+n, grads[n], g, **cfg["precision"]["paired_head_gradient_allclose"])
            for n in a0:
                a, g, actual = (x[n].astype(np.float64) for x in (a0, grads, after))
                require(case+".shape."+n, a.shape == g.shape == actual.shape)
                lr = opt["head_actual_lr"] if n.startswith("heads.") else opt["adapter_actual_lr"]
                expected = a*(1-lr*opt["weight_decay"])-lr*g/(np.abs(g)+opt["eps"])
                tol = cfg["precision"]["manual_AdamW_parameter_oracle"]
                close(case+".AdamW_oracle."+n, actual, expected, atol=tol["atol"], rtol=tol["rtol"])
                if n.endswith((".aq", ".av")):
                    check(case+".A_gradient_exact0."+n, np.count_nonzero(g) == 0)
                if n.endswith((".bq", ".bv")):
                    check(case+".B_gradient_and_step_nonzero."+n, np.count_nonzero(g) > 0 and np.count_nonzero(actual) > 0)
                compare(case+".parameter_record."+n, r["parameter_records"][n], {
                    "gradient_norm_CPU_FP64": norm(g), "gradient_nonzero": int(np.count_nonzero(g)),
                    "gradient_absmax": float(np.max(np.abs(g))), "gradient_fraction_abs_le_epsilon": float(np.mean(np.abs(g) <= opt["eps"])),
                    "parameter_delta_norm_CPU_FP64": norm(actual-a),
                    "AdamW_FP64_oracle_max_abs_error": float(np.max(np.abs(actual-expected))), "actual_lr": lr})
            merged = []
            for slot, block in enumerate(cfg["blocks"][layout]):
                prefix = f"encoder.layers.encoder_layer_{block}.self_attention.parametrizations.in_proj_weight.0"
                for qv in ("q", "v"):
                    a, b = after[prefix+".a"+qv].astype(np.float64), after[prefix+".b"+qv].astype(np.float64)
                    merged.append({"slot": slot, "projection": qv, "CPU_FP64_frobenius": norm(b@a)})
            compare(case+".merged_QV", r["merged_QV_updates"], merged)
            joint = float(np.sqrt(sum(x["CPU_FP64_frobenius"]**2 for x in merged)))
            require(case+".nonzero_merged_and_initial_features", joint > 0 and rms(f0) > 0)
            f1, = features(d/"features.npz", ["post_step_features"])
            s, sp = [rms(f1[j]-f0[j]) for j in range(2)], rms(f1-f0)
            resolved = [s[j] > 10*floors[j] for j in range(2)]
            all_resolved = all_resolved and all(resolved)
            rows.append({"layout": layout, "gradient_batch": batch, "probe0_S": s[0], "probe1_S": s[1],
                         "pooled_S": sp, "relative_pooled_S": sp/rms(f0), "probe0_resolved": resolved[0], "probe1_resolved": resolved[1],
                         "merged_QV_joint_Frobenius": joint, "S_per_merged_QV_Frobenius": sp/joint})
            scales[layout].append(s); pooled[layout].append(sp)
            norm_diagnostics.append({"case": case, "merged_QV": merged})
        fr = readj(folder/"frozen-base-receipt.json")
        compare(layout+".frozen_receipt", fr, {"unchanged": True, "initial_matches_original_pilot": True, "initial_slot_sha256": state_hash(a0, False)})
        check(layout+".frozen_hash_receipt_equal", fr["before_sha256"] == fr["after_sha256"] and bool(HEX.fullmatch(fr["before_sha256"])))
    check("paired_by_slot_initial_values", state_hash(init["front"], False) == state_hash(init["back"], False))
    close("initial_front_back_features", fzero["front"], fzero["back"], **cfg["precision"]["initial_front_back_allclose"])
    front, back = np.asarray(pooled["front"]), np.asarray(pooled["back"])
    positive = bool(np.all(front > 0) and np.all(back > 0) and np.all(np.asarray(scales["front"]) > 0) and np.all(np.asarray(scales["back"]) > 0))
    ratios = front/back if positive else None
    pr = np.asarray(scales["front"])/np.asarray(scales["back"]) if positive else None
    variation = {}
    for layout, v in pooled.items():
        variation[layout] = {"all4_pooled_S": v, "mean_S": float(np.mean(v)), "sample_SD_S": float(np.std(v, ddof=1)),
            "CV_S": float(np.std(v, ddof=1)/np.mean(v)) if np.mean(v) > 0 else None,
            "sample_SD_logS": float(np.std(np.log(v), ddof=1)) if min(v) > 0 else None, "range_S": [float(min(v)), float(max(v))]}
    logsd = max(v["sample_SD_logS"] for v in variation.values()) if positive else None
    recomputed = {"rows": rows, "baseline_precision": baselines, "all8_case_two_probe_changes_resolved": all_resolved,
        "paired_front_over_back_ratios": ratios.tolist() if positive else None, "paired_per_probe_ratios": pr.tolist() if positive else None,
        "geometric_mean_front_over_back": float(np.exp(np.mean(np.log(ratios)))) if positive else None,
        "paired_ratio_range": [float(ratios.min()), float(ratios.max())] if positive else None, "batch_variation": variation,
        "consistent_direction_all4_batches_and_both_probes": bool(all_resolved and positive and (np.all(pr > 1) or np.all(pr < 1))),
        "absolute_mean_log_ratio_divided_by_larger_within_layout_log_SD": float(abs(np.mean(np.log(ratios)))/logsd) if positive and logsd > 0 else None,
        "new_augmentation_interaction_or_novel_contribution_established": False, "original_pilot_engineering_gate_changed": False,
        "full_training_LR_search_test_reads_launched": False}
    reported = readj(root/"analysis.json")
    compare("saved_analysis", reported, recomputed)
    return {"independently_recomputed": recomputed, "merged_norm_diagnostics": norm_diagnostics,
            "evidence_limits": ["No forward/backward/optimizer replay: saved gradients and frozen-base hash receipts are checked, not regenerated.",
                "Input tensor digests are checked for identity/format; original input tensors are not saved, so pixel-transform reproduction is not independently verified.",
                "Compressed-image hashes cover only96 selected retained training members; no image decoding, dev logits or reserved/test members.",
                "UTC order is the saved host declaration; no external timestamp attestation. Numerical floor is a predeclared empirical rule."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("independent-verification.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior independent audit: "+str(args.output))
    report = {"at_utc": datetime.now(timezone.utc).isoformat(), "root": str(args.root.resolve()),
              "verifier_sha256": sha(Path(__file__).resolve()), "CPU_threads": torch.get_num_threads(),
              "scope": "Saved-artifact CPU recomputation only; no model imports, forward, autograd, optimizer or GPU access."}
    error = None
    try:
        report.update(audit(args.root.resolve()))
    except Exception as e:
        error = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    failed = [x for x in EVENTS if not x["passed"]]
    report.update(status="passed_saved_artifact_recomputation" if error is None and not failed else "failed_or_incomplete_preserve_all",
                  checks=len(EVENTS), failed_checks=failed, events=EVENTS, inputs=INPUTS, error=error,
                  versions={"numpy": np.__version__, "torch": torch.__version__, "python": sys.version})
    with args.output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(EVENTS), "failed_checks": len(failed), "output": str(args.output)}))
    return 0 if error is None and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
