# Analytical notes — no empirical claim

## 1. Opportunity and agreement are different variables

For a fixed candidate box d and an explicitly specified view policy pi, C_pi(d) is the geometrically eligible view set; K_pi(d)=|C_pi(d)|. Eligibility uses the published CAR margin 0.5 sqrt(w h), waived at image boundaries. M_pi(d) counts eligible views with at least one same-class box at IoU >=0.75. It counts views, not boxes. A_pi(d)=M_pi(d)/K_pi(d). K must include eligible views with **zero detections**, so it cannot be reconstructed merely from the view IDs appearing in the prediction rows.

If the candidate's own source view is eligible, self-match contributes one. For K>=1,

    A = [1 + sum_{j != source} I_j] / K.

Conditioning on K and on that source eligibility,

    E[A | K, source eligible] = 1/K + (1/K) sum q_j,

where q_j is the conditional probability of a match in another eligible view. Equality does not require independence. If all q_j equal q, this reduces to 1/K + (1-1/K)q. When K=1 and the source is eligible, A=1 regardless of correctness. If the source is **not** eligible, this argument does not apply. This selection/self-match identity is elementary, not a new result. CAR already investigated own-view exclusion; deleting it is not proposed as a new method.

For exchangeable other-view indicators with variance q(1-q) and common correlation rho,

    Var(A | K, source eligible) = (K-1) q(1-q) [1+(K-2)rho] / K^2.

This is an illustrative conditional model, not an assumption about actual crops. There is no guarantee that q or rho stays constant when geometry changes. A difference in K alone does not identify a causal effect: crop context, visibility, matching jitter and candidate selection also change.

## 2. Max score is not a probability guarantee

For a **pre-specified** background hypothesis with scores S_j, the event that max S_j exceeds t is a union. Its probability is bounded by max_j P(S_j>t) and min(1,sum_j P(S_j>t)). The independent equal-marginal expression 1-(1-p)^K is only an illustration. It must not be applied as if correlated overlapping crops were independent, or as if a selected merged box were pre-specified.

SAHI Greedy-NMM takes maximum pair score, but also unions box coordinates and changes clusters. The proposal does not derive its error rate from the maximum alone. It uses class-wise NMS to align with the CAR comparator, not a silent mixture of NMS and Greedy-NMM.

## 3. Fix score units before discussing transferable thresholds

CAR's `car_scores` returns t=s(1+5A). Its `normalize_scores` divides by the maximum of the supplied score array. Positive global rescaling preserves the array's rank, but the divisor depends on what array was supplied. A source threshold expressed after such per-array scaling is not automatically portable.

For the declared reliability reference, use u=t/6, a fixed constant map into [0,1] for s,A in [0,1]. This is rank-equivalent to CAR on any fixed pooled set in exact arithmetic, but it is **not** presented as the authors' native probability output. It removes an avoidable normalization-context artifact from the scientific question. Ties/float32 changes require checks before interpreting any result. u is a score, not a calibrated probability by construction.

## 4. Strict monotonic calibration does not solve this operational question by itself

If f is a single strictly increasing score map and the source operating threshold is f(tau), then

    {d: f(s(d)) >= f(tau)} = {d: s(d) >= tau}.

The accepted set is identical on every target policy too. Such calibration cannot improve FP/recall at a source-selected operating point merely by changing score units. Isotonic plateaus require a declared tie rule. Class- or feature-conditional maps can reorder candidates and are not covered by this equivalence; CAR's released 13-feature baseline already does this.

## 5. Geometry-only thinning is a declared reference policy

Let the dense CAR tile starts be sorted lists X and Y (S=512, r=0.6). Retain every second start beginning with index 0, plus the last start on each axis, then keep their Cartesian product and the full-image view. Consecutive retained starts differ by at most 2 floor(512*0.4)=408 <512, so coverage is retained for ordinary axis starts; border handling must still be checked on actual size metadata. Interior overlap is about 20.31%, **not an exact native r=0.2 SAHI grid**.

Only published predictions for retained views are used in the sparse operational arm. This is a retrospective estimate of deterministic inference with skipped views, not a measured speedup. Pixel arrays and predictions of retained views are unchanged. Candidate membership and the set of votes change; they are not held equal. A separate fixed-candidate diagnostic can isolate vote-set changes from candidate selection, but not K from all correlations in the votes.
