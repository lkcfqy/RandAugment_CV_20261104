"""Separate CPU-only audit of a stopped diagnostic; does not repair its gates.

Only saved small checkpoints, training-probe arrays and provenance are read.
No imports from training/model modules, image decoding, optimizer, forward,
autograd, GPU queries, extra cases, new thresholds or original outcomes.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYTORCH_ENABLE_MPS_FALLBACK="0")
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import traceback
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

WORK = Path(__file__).resolve().parent
PILOT = Path("/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/pilot-20261022")
FULL_VERIFIER_SHA = "0f50586e5a96c4c3fa7878b9183140af18f8f089644bce9634ac0837f7a0960e"
EVENTS, INPUTS = [], {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    value = h.hexdigest()
    INPUTS[str(path)] = {"sha256": value, "bytes": Path(path).stat().st_size}
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def check(name, passed, kind="artifact_consistency", **detail):
    EVENTS.append({"name": name, "kind": kind, "passed": bool(passed), **detail})
    return bool(passed)


def require(name, passed):
    if not check(name, passed):
        raise ValueError(name)


def close(name, actual, expected, atol=1e-15, rtol=2e-12):
    a, e = np.asarray(actual), np.asarray(expected)
    ok = a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all()
    return check(name, ok and np.allclose(a, e, atol=atol, rtol=rtol),
                 max_abs_difference=float(np.max(np.abs(a-e))) if ok else None, atol=atol, rtol=rtol)


def state(path, key):
    sha(path)
    meta = torch.load(path, map_location="cpu", weights_only=True)
    require(str(path)+".mapping", isinstance(meta.get(key), dict))
    arrays = {}
    for n, t in meta[key].items():
        require(str(path)+":"+n+".finite_CPU_FP32", isinstance(t, torch.Tensor)
                and t.device.type == "cpu" and t.dtype == torch.float32 and bool(torch.isfinite(t).all()))
        arrays[n] = t.detach().contiguous().numpy()
    return meta, arrays


def statehash(state, names=True):
    h = hashlib.sha256()
    for n, a in state.items():
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
        require(str(path)+":"+k+".finite_shape_dtype", a.shape == (2,16,768)
                and a.dtype == np.float32 and np.isfinite(a).all())
    return [a.astype(np.float64) for a in values]


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64)**2)))


def norm(a):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)))


def encode_numpy_scalar(value):
    # Preserve cached scalar values and all calculations; convert JSON types only.
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported JSON value: "+type(value).__name__)


def audit(root):
    cfg, lock = read(root/"protocol.json"), read(root/"protocol-lock.json")
    require("protocol_hash", sha(root/"protocol.json") == lock["protocol_sha256"])
    for name, value in {**lock["code_sha256"], **lock.get("supporting_files_sha256", {})}.items():
        require("frozen_file_hash."+name, Path(name).name == name and sha(root/name) == value)
    require("original_common_hash", sha(PILOT/"common.py") == lock["original_common_sha256"])
    require("original_delivery_manifest_hash", sha(PILOT/"delivery-manifest.json") == cfg["protected_original_pilot_delivery_manifest_sha256"])
    require("full_verifier_preserved", sha(WORK/"verify_independent.py") == FULL_VERIFIER_SHA
            and sha(root/"verify_independent.py") == FULL_VERIFIER_SHA)
    full = read(WORK/"independent-verification.json")
    require("full_verifier_failure_preserved", full["status"] == "failed_or_incomplete_preserve_all"
            and full["error"]["type"] == "FileNotFoundError" and "results/completed.json" in full["error"]["message"]
            and full["verifier_sha256"] == FULL_VERIFIER_SHA and not full["failed_checks"])
    require("unchanged_oracle_tolerance", cfg["precision"]["manual_AdamW_parameter_oracle"]["atol"] == 2e-10
            and cfg["precision"]["manual_AdamW_parameter_oracle"]["rtol"] == 2e-6)
    require("unchanged_head_tolerance", cfg["precision"]["paired_head_gradient_allclose"] == {"atol": 2e-8, "rtol": 2e-5})
    opt = cfg["optimizer"]
    require("original_first_warmup_step", opt["adapter_actual_lr"] == .0003/111 and opt["head_actual_lr"] == .001/111
            and opt["weight_decay"] == .01 and opt["eps"] == 1e-8 and opt["betas"] == [.9,.999])
    require("retained_training_only96_selection", len(cfg["selected_rows"]) == 96
            and len({r["source_id"] for r in cfg["selected_rows"]}) == 96
            and all(r["candidate_partition"] == "train" for r in cfg["selected_rows"]))
    started, failure = read(root/"results/started.json"), read(root/"diagnostic-failure.json")
    require("stopped_at_back0_head_gate", failure["phase"] == {"stage": "paired_head_and_optimizer_oracle", "layout": "back", "gradient_batch": 0}
            and "paired_head_grads" in failure["traceback"] and failure["no_automatic_retry_or_parameter_change"] is True)
    check("no_completed_marker", not (root/"results/completed.json").exists())
    check("no_main_analysis_or_back_probe", not (root/"analysis.json").exists() and not (root/"results/back/batch0/features.npz").exists())
    check("only5_post_step_states", sorted(str(p.relative_to(root/"results")) for p in (root/"results").glob("*/batch*/post-step-trainable.pt"))
          == ["back/batch0/post-step-trainable.pt"]+[f"front/batch{i}/post-step-trainable.pt" for i in range(4)])
    check("only4_completed_case_receipts", sorted(str(p.relative_to(root/"results")) for p in (root/"results").glob("*/batch*/receipt.json"))
          == [f"front/batch{i}/receipt.json" for i in range(4)])
    parse_time = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    check("declared_freeze_start_failure_order", parse_time(cfg["at_utc"]) <= parse_time(lock["at_utc"]) < parse_time(started["at_utc"]) < parse_time(failure["at_utc"]))
    initials, fzero, repeats, initial_info = {}, {}, {}, {}
    for layout in ("front", "back"):
        folder = root/"results"/layout
        imeta, init = state(folder/"initial-trainable.pt", "trainable_state")
        _, old = state(PILOT/"results"/("front-weak" if layout == "front" else "back-weak")/"initial-trainable-checkpoint.pt", "trainable_state")
        require(layout+".same_original_names", set(init) == set(old) and len(init) == 14)
        check(layout+".65317_parameters", sum(a.size for a in init.values()) == 65317)
        for n in init:
            check(layout+".original_initial."+n, np.array_equal(init[n], old[n]))
            if n.endswith((".bq", ".bv")):
                check(layout+".B0."+n, np.count_nonzero(init[n]) == 0)
        check(layout+".initial_protocol", imeta["protocol_sha256"] == lock["protocol_sha256"])
        initials[layout] = init
        f0, rep = features(folder/"initial-features.npz", ["features", "repeat_features"])
        fzero[layout], repeats[layout] = f0, rep
        initial_info[layout] = {"slot_hash": statehash(init, False), "probe_RMS": [rms(f0[j]) for j in range(2)],
            "repeat_noise_RMS": [rms(rep[j]-f0[j]) for j in range(2)],
            "floors": [max(rms(rep[j]-f0[j]), np.finfo(np.float32).eps*rms(f0[j])) for j in range(2)]}
    check("initial_slot_tensors_identical", statehash(initials["front"], False) == statehash(initials["back"], False))
    close("initial_features_protocol_gate", fzero["front"], fzero["back"], **cfg["precision"]["initial_front_back_allclose"])
    initial_info["front_back_features_exact_equal"] = bool(np.array_equal(fzero["front"], fzero["back"]))
    initial_info["front_back_feature_max_abs_difference"] = float(np.max(np.abs(fzero["front"]-fzero["back"])))
    cases, front_scales, merged_arrays, grad0, losses = [], [], [], {}, {}
    for layout, batch in [("front", i) for i in range(4)]+[("back", 0)]:
        case = f"{layout}-batch{batch}"
        d, init = root/"results"/layout/f"batch{batch}", initials[layout]
        gm, grads = state(d/"gradients.pt", "gradients")
        pm, after = state(d/"post-step-trainable.pt", "trainable_state")
        require(case+".names", set(init) == set(grads) == set(after))
        check(case+".case_protocol", gm["case"] == pm["case"] == case and gm["protocol_sha256"] == pm["protocol_sha256"] == lock["protocol_sha256"])
        loss = read(d/"pre-step-loss.json")
        losses[case] = float(loss["CE_repr"])
        check(case+".finite_saved_loss", loss["finite"] is True and np.isfinite(losses[case]))
        receipt = read(d/"receipt.json") if layout == "front" else None
        records, all_oracle, zero_a, nonzero_b = {}, True, 0, 0
        for n in init:
            a, g, actual = (x[n].astype(np.float64) for x in (init, grads, after))
            require(case+".shape."+n, a.shape == g.shape == actual.shape)
            lr = opt["head_actual_lr"] if n.startswith("heads.") else opt["adapter_actual_lr"]
            expected = a*(1-lr*opt["weight_decay"])-lr*g/(np.abs(g)+opt["eps"])
            passed = close(case+".fixed_AdamW_oracle."+n, actual, expected, atol=2e-10, rtol=2e-6)
            all_oracle = all_oracle and passed
            if n.endswith((".aq", ".av")):
                z = np.count_nonzero(g) == 0
                zero_a += int(z); check(case+".A_data_gradient_exact0."+n, z, kind="protocol_gate")
            if n.endswith((".bq", ".bv")):
                nz = np.count_nonzero(g) > 0
                nonzero_b += int(nz); check(case+".B_data_gradient_nonzero."+n, nz, kind="protocol_gate")
            rec = {"gradient_norm_CPU_FP64": norm(g), "gradient_nonzero": int(np.count_nonzero(g)),
                "gradient_absmax": float(np.max(np.abs(g))), "gradient_fraction_abs_le_epsilon": float(np.mean(np.abs(g) <= opt["eps"])),
                "parameter_delta_norm_CPU_FP64": norm(actual-a), "AdamW_FP64_oracle_max_abs_error": float(np.max(np.abs(actual-expected))), "actual_lr": lr}
            records[n] = rec
            if receipt:
                for key, value in rec.items():
                    close(case+".saved_parameter_record."+n+"."+key, receipt["parameter_records"][n][key], value)
        merged = []
        for slot, block in enumerate(cfg["blocks"][layout]):
            prefix = f"encoder.layers.encoder_layer_{block}.self_attention.parametrizations.in_proj_weight.0"
            for qv in ("q", "v"):
                a, b = after[prefix+".a"+qv].astype(np.float64), after[prefix+".b"+qv].astype(np.float64)
                value = norm(b@a)
                merged.append({"slot": slot, "projection": qv, "CPU_FP64_frobenius": value})
                if receipt:
                    close(case+f".saved_merged.{slot}.{qv}", receipt["merged_QV_updates"][len(merged)-1]["CPU_FP64_frobenius"], value)
        joint = float(np.sqrt(sum(x["CPU_FP64_frobenius"]**2 for x in merged)))
        merged_arrays.append([r["CPU_FP64_frobenius"] for r in merged])
        record = {"case": case, "A_zero_gradient_tensors_of6": zero_a, "B_nonzero_gradient_tensors_of6": nonzero_b,
                  "all14_parameter_oracles_pass": all_oracle, "parameters": records, "merged_QV": merged,
                  "joint_merged_QV_Frobenius": joint, "saved_training_CE": losses[case], "post_step_probe_available": layout == "front"}
        if layout == "front":
            check(case+".reset_receipt", receipt["reset_from_initial_sha256"] == statehash(init)
                  and receipt["optimizer_steps_in_this_fresh_state"] == 1)
            close(case+".loss_receipt", receipt["training_batch_CE_before_step"], losses[case])
            f1, = features(d/"features.npz", ["post_step_features"])
            s, sp = [rms(f1[j]-fzero[layout][j]) for j in range(2)], rms(f1-fzero[layout])
            floors = initial_info[layout]["floors"]
            record["feature_change"] = {"probe_S": s, "pooled_S": sp, "relative_pooled_S": sp/rms(fzero[layout]),
                "floor": floors, "S_over_floor": [s[j]/floors[j] for j in range(2)],
                "resolved_gt10_floor": [s[j] > 10*floors[j] for j in range(2)], "S_per_joint_merged_Frobenius": sp/joint}
            front_scales.append([s[0], s[1], sp])
        if batch == 0:
            grad0[layout] = {n: a.astype(np.float64) for n, a in grads.items() if n.startswith("heads.")}
        cases.append(record)
    fr = read(root/"results/front/frozen-base-receipt.json")
    check("front_frozen_hash_receipt", fr["unchanged"] is True and fr["before_sha256"] == fr["after_sha256"])
    check("back_frozen_receipt_unavailable", not (root/"results/back/frozen-base-receipt.json").exists())
    head, masks, deltas, tolerances = {}, {}, {}, {}
    for n, ref in grad0["front"].items():
        actual = grad0["back"][n]
        delta, tolerance = actual-ref, 2e-8+2e-5*np.abs(ref)
        outside = np.abs(delta) > tolerance
        passed = bool(np.allclose(actual, ref, atol=2e-8, rtol=2e-5))
        check("paired_head_gradient."+n, passed, kind="protocol_gate", outside_elements=int(outside.sum()), total_elements=ref.size)
        head[n] = {"predeclared_atol": 2e-8, "predeclared_rtol": 2e-5, "passed": passed,
            "outside_elements": int(outside.sum()), "total_elements": ref.size,
            "max_abs_difference": float(np.max(np.abs(delta))), "L2_difference": norm(delta),
            "relative_L2_to_front": norm(delta)/norm(ref), "front_L2": norm(ref), "back_L2": norm(actual),
            "max_abs_difference_divided_by_elementwise_tolerance": float(np.max(np.abs(delta)/tolerance))}
        masks[n], deltas[n], tolerances[n] = outside, delta, tolerance
    values = np.asarray(front_scales)[:,2]
    front_variation = {"all4_pooled_S": values.tolist(), "mean": float(np.mean(values)), "sample_SD": float(np.std(values, ddof=1)),
                       "CV": float(np.std(values, ddof=1)/np.mean(values)) if np.mean(values) > 0 else None,
                       "sample_SD_logS": float(np.std(np.log(values), ddof=1)) if np.min(values) > 0 else None}
    np.savez_compressed(WORK/"partial-recomputed-arrays.npz", front_S_probe0_probe1_pooled=np.asarray(front_scales),
        front_floors=np.asarray(initial_info["front"]["floors"]), merged_QV_Frobenius_5cases=np.asarray(merged_arrays),
        head_weight_difference=deltas["heads.head.weight"], head_weight_original_tolerance=tolerances["heads.head.weight"],
        head_weight_outside_original_tolerance=masks["heads.head.weight"], head_bias_difference=deltas["heads.head.bias"])
    return {"protocol_sha256": lock["protocol_sha256"], "full_verifier_checks_before_missing_completed": full["checks"],
        "saved_steps_executed": 5, "completed_case_receipts": 4, "cases": cases, "initialization_and_precision": initial_info,
        "paired_head_gradients_batch0": head, "saved_losses": losses,
        "back_minus_front_batch0_CE": losses["back-batch0"]-losses["front-batch0"], "front_only_batch_variation": front_variation,
        "full_front_back_S_comparison": None, "scientific_negative_established": False,
        "finish_decision": "Stop. Diagnostic incomplete because the original paired-head gate failed; no resumption, tolerance change, extra case, forward or GPU.",
        "limitations": ["Gradients/losses are saved observations, not regenerated. No gradient-batch activations/logits were saved to locate the mismatch.",
                        "FP32/autograd route differences are plausible, not established; exact clean-probe equality does not prove gradient-batch equality.",
                        "Back probe and back frozen-base receipt are absent. No front/back feature ratio, direction or mechanism conclusion is available.",
                        "Only fixed training photos/probes; no development accuracy, test read, seed inference or original-pilot outcome revision."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    output = WORK/"partial-audit.json"
    require("no_partial_audit_overwrite", not output.exists() and not (WORK/"partial-recomputed-arrays.npz").exists())
    report = {"at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__).resolve()),
              "root": str(args.root.resolve()), "CPU_threads": torch.get_num_threads(), "GPU_forward_autograd_or_optimizer_executed": False}
    error = None
    try:
        report.update(audit(args.root.resolve()))
    except Exception as e:
        error = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    artifact_failures = [x for x in EVENTS if not x["passed"] and x["kind"] == "artifact_consistency"]
    gate_failures = [x for x in EVENTS if not x["passed"] and x["kind"] == "protocol_gate"]
    report.update(status="partial_artifacts_recomputed_protocol_gate_failure_preserved" if error is None and not artifact_failures else "partial_audit_error_preserve_all",
                  diagnostic_status="incomplete_not_scientific_negative", artifact_failures=artifact_failures,
                  protocol_gate_failures=gate_failures, error=error, checks=len(EVENTS), events=EVENTS, inputs=INPUTS,
                  versions={"python": sys.version, "numpy": np.__version__, "torch": torch.__version__})
    with output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False, default=encode_numpy_scalar); f.write("\n")
    print(json.dumps({"status": report["status"], "diagnostic_status": report["diagnostic_status"], "checks": len(EVENTS),
                      "artifact_failures": len(artifact_failures), "protocol_gate_failures": len(gate_failures), "error": error,
                      "output": str(output)}, ensure_ascii=False))
    return 1 if error is not None or artifact_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
