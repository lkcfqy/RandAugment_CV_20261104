# Source and controller boundary

Examined [arXiv:2510.00434v1](https://arxiv.org/abs/2510.00434v1), submitted 2025-10-01 02:26:52 UTC; the arXiv history showed only v1 at retrieval. Saved HTML/PDF bytes and SHA-256 hashes are in `sources.json`. PDF has 12 pages; equations were visually checked on pages 5–6 (`sources/paper-v1-page-5.png`, `sources/paper-v1-page-6.png`).

For adjacent states θ₋, θ₊ and predictions p₋, p₊, §3 gives:

| Source | Mathematical content |
|---|---|
| Eq. 2 | `abs(g_current · (theta_previous-theta_current))`; no unit-direction denominator |
| Eq. 3 | switches to `g_previous` and introduces `1/eta`; first-order absolute loss change divided by eta |
| Eq. 4 | printed signed difference has the opposite sign from standard negative-log CE; subsequent absolute values erase that sign |
| Eq. 5 | `KL(p_current || p_previous)` after explicitly replacing the label with the current prediction |
| Eq. 6 | `sum((abs(score)-window_mean)^2)`; no division by L or L−1 |
| Eq. 7 | `EMA_new = beta * window_statistic + (1-beta) * EMA_old` |

The paper interprets this as optimization influence, and explicitly discloses the prediction substitution. It is therefore a heuristic proxy, not a concealed exact identity. Adjacent steps and epoch windows both appear. Introduction gives L=5 as an example; §4.8 varies L and beta, without a recovered default specification.

**Unrecovered:** scoring input view, optimizer/epoch timing, actual learning-rate normalization, warm-up, EMA initialization/default beta, normalization population/ties/epsilon, explicit inversion, magnitude discretization and operation table. Text says lower variance receives stronger augmentation but supplies no complete mapping; the cited appendix is unresolved in this v1.

**Code/access search:** No author implementation link occurs in the saved paper; focused title, arXiv-ID, SADA/author and author-account searches found no attributable official repository. This is bounded non-discovery, not proof of absence. Unrelated repositories named SADA were excluded. `https://openreview.net/forum?id=O0OPGytwpb` returned a browser-verification challenge; no bypass or alternate endpoint was attempted. The accessible arXiv version supports this formula audit; a newer OpenReview controller was not verified.

**Consequences for this audit:** no actual augmentation actions are generated. EMA uses declared zero initialization and four sensitivity coefficients, not claimed author defaults. All metrics receive the same constant eta divisor within each comparison. A changed eta schedule/trajectory is not equivalent to rescaling a fixed trajectory.
