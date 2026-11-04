1. Freeze data/config/weights and CPU controls. Reuse the verified passive pipeline observer.
2. Run only A iteration2 and warmed M iterations43/44 full train steps in separate processes, within600s total.
3. Gate full paired workload with measured time/RSS, geometric upper-bound factor,25% margin, both complete model saves and128 CPU dev forwards;8GiB combined storage and2h total.
4. If PASS, discard preflight process/state and independently run A then M seed20261031 U200. Compare complete initial state/optimizer/trainable/anchor controls.
5. Both complete final checkpoints must pass safe readback before any dev inference. Evaluate128 unique records using existing CPU/COCO implementation; independently recompute persisted metrics.
6. Preserve real source/operator counts, timing/RSS and all checkpoint/record hashes. No cloud, uploads, installs or test.
