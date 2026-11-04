FULL_RAW_PR_STEPS.npz is a NumPy compressed numeric-only archive. Open with numpy.load(path, allow_pickle=False).

Keys A_cap100_*, A_cap300_*, B_cap100_*, B_cap300_* each contain: score and dt_id (length64*cap), TP_cumulative, FP_cumulative, precision and recall (10 x64*cap). Columns follow descending score, stable image_id then within-image original order. dt_id is original prediction JSON index+1. iou_thresholds and recall_grid are supplied. All-area, one-category, original187GT; no score threshold or reassignment. The raw curve is non-interpolated. FULL_INTERPOLATED_PR_CURVES.csv retains all101 official recall points for both caps and all4 area ranges; ALL_IOU_BINS.csv also includes every group-scale.

Zero score_at_recall may denote an unreached recall point, following pycocotools; it is not a newly chosen operating threshold. N/A CSV cells mean no GT.
