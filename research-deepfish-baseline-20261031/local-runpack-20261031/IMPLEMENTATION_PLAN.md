# DeepFish Minimal Local Run Package Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; a fresh reviewer checks the finished package. No model/GPU/upload execution is authorized.

**Goal:** Prepare one auditable, <=13 MiB upload archive for the fixed 5-shot A/B experiment, with non-model tests and a single future command.
**Architecture:** Two complete JSON configurations use the same official MMDetection GroundingDINO Swin-T and standard geometry. A small stdlib controller prepares a pinned candidate environment/assets only after explicit runtime authorization; isolated official-Runner workers perform the two preflights and two formal arms. Pure contract/lifecycle functions remain testable without models.
**Tech Stack:** Python stdlib, existing local NumPy/OpenCV for artificial tests; future official MMDetection 3.3.0, PyTorch 2.0.0/cu117, MMCV 2.1.0.
**Spec:** Parent delegation on 2026-10-31; previous `followup-group-provenance-20261031/reports/MINIMAL_TWO_ARM_PLAN_ZH.md` and frozen DEV_SELECTION.json.

## Global Constraints

- Local preparation only: no installation, upload, GPU assignment, model construction or inference now.
- One official 5-shot episode, seed 20261031; A/B Color(prob=1, level=0), factor 1.0/0.8, cv2 uint8 BGR.
- 69 original public JPEGs; existing support/dev records unchanged, especially annotations 4195/4660 xywh/area. No official test in archive.
- U=5E, each support image exactly E formal exposures. Each preflight is 15 updates and separately accounted; all its model/optimizer state discarded.
- U selected once from timing alone before any dev/test metric, final checkpoint only, no save_best.
- Future ceilings: 2 CU and 2 allocated-GPU hours including preparation/finalization; absent compatible interpreter/rate/persistence/release evidence means RESOURCE_UNDETERMINED.
- All external assets downloaded in future Colab from fixed official URLs and known SHA; weights_only=True, no unsafe fallback/trust_remote_code.
- Upload archive <=13 MiB, no full repository/weights/old experiment/personal paths/credentials. No Git commit/push is part of this local artifact delivery.

## Review Focus

1. Accidental execution without approval/context must fail before imports, network, installation or model construction.
2. Color RNG or geometry drift must abort paired comparison; random state and GT traces must be comparable.
3. Rate, deadline or persistence ambiguity must block rather than silently reduce the 69-image protocol.
4. Preflight state reuse, unequal exposures or intermediate-score selection must fail contract checks.
5. Package membership, altered edge boxes, private metadata or omitted hashes must fail packaging checks.

## Task 1 — Pure contracts, lifecycle and tests

Files: `payload/contracts.py`, `payload/lifecycle.py`, `tests/test_contracts.py`, `tests/test_lifecycle.py`.
Interfaces: `choose_updates(timings, remaining_seconds) -> {updates, estimate_seconds}`; `verify_exposures(ids, updates)`; atomic evidence and finite release helpers.
- [x] Write behavioral tests first: unknown rates, exhausted budget, monotone timing selection with no loss input, U multiple-of-five exposures, atomic persistence and uncertain release.
- [x] Observe RED, implement only the required contracts, observe GREEN.
- [x] Record the exact test command/result and reused lifecycle source provenance in progress.md.

## Task 2 — Explicit configurations and minimal driver

Files: `payload/config_A.json`, `payload/config_B.json`, `payload/driver.py`, `payload/model_worker.py`, candidate dependency/asset pins.
Interfaces: driver accepts one external `--run-context` JSON, rejects missing authorization, and launches isolated subprocess phases; worker records initialization, exact source/geometry/GT/RNG traces and tensor-only final state.
- [x] Add artificial/static tests for Color identity/RNG parity, one source exposure cycle, exact config differences, evaluation resize→annotations, safe loading and no-autostart.
- [x] Implement official Color/pipeline runtime gate, phase isolation and deterministic U rule. Keep core model imports inside the authorized worker.
- [x] Reuse only atomic persistence, finite expiry and ownership/reconciliation semantics from prior Colab code; remove paths/credentials/history.
- [x] Run non-model tests; do not claim CUDA or full dependency compatibility verified.

## Task 3 — Data bundle, privacy, final review

Files: 69 `payload/data/images/*.jpg`, support/dev JSON, README, full SHA manifest; `deepfish_5shot_two_arm.zip` plus local reports.
- [x] Copy exact selected archive members after checking original hashes; serialize unchanged annotation records.
- [x] Validate 5+64 images, 5+187 boxes, unchanged edge boxes, no test members, no JPEG personal metadata, relative paths and exact allowed file list.
- [x] Build reproducible zip, enforce <=13 MiB, verify every member/hash and record archive bytes/SHA.
- [x] Run all package non-model tests, one fresh static review, fix important findings and re-run affected checks.
- [x] Deliver one future command and explicit pending authorization/resource status; no upload/run.

## Self-review

All parent constraints map to Tasks 1–3. Shared interfaces are timing-only profile→U freeze, config→worker trace, and exact package manifest→driver verification. No missing dependency on model execution is hidden: local tests use synthetic arrays and AST/config checks; official CUDA imports/Color gate remain future runtime checks. Staging is isolated from historical files, then appended to the existing research directory. The skill's extra plan-approval/commit/worktree ceremony is superseded by the explicit instruction to self-review then prepare this local artifact without extending the framework.
