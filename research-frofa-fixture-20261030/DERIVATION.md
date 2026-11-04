# Fixed-feature conditional controls (pre-run derivation)

All expectations condition on one fixed sample's X and a. Flatten in channel-major order. Each channel has independent randomness, so the full conditional covariance is block diagonal across channels, not a covariance across data samples. There is no apply gate. This is a fixture for the assigned mathematical mechanism, not a novelty claim.

For one channel let l=min x, h=max x, d=h-l+1e-8, u=(x-l)/d and g_t(s)=d[clip(u_t+s,0,1)-u_t]. C uses one s~Uniform[-a,a] per channel. E* replaces s with mutually independent s_t of the same law, keeping all other operations unchanged. For a=0 both are the point mass at zero. Constants have d=1e-8, u=0: positive shifts survive clipping, so their perturbations are small but not identically zero. The upper reconstruction boundary is h+epsilon, not exactly h.

## Exact moments and finite factor

Sort and deduplicate {-a,a} and all breakpoints -u_t,1-u_t strictly inside [-a,a]. On each interval [L,R], g is affine. Put m=(L+R)/2, r=(R-L)/2, nodes q=m±r/sqrt(3), weights w=(R-L)/(4a) for each node. These normalized weights sum to one. Two-node Gauss-Legendre integrates polynomials through degree three, hence each g_t, g_t g_s, and each centered cubic product exactly in real arithmetic.

Let mu=sum_j w_j g(q_j) and B[:,j]=sqrt(w_j)[g(q_j)-mu]. Then BB^T=Sigma=E[(g-mu)(g-mu)^T]. Thus G_C=mu+Bz, z~N(0,I), has the entire required conditional covariance; no eigenvalue clipping, Cholesky, covariance shrinkage, second clamp, or renormalization is involved. Equality is algebraic in real arithmetic, verified to recorded float64 tolerances. Independent piecewise-polynomial integration is an oracle, not Monte Carlo.

E* has the identical full scalar marginal at every coordinate because the deterministic map and scalar input law are identical. Its covariance is diag(Sigma_C). G_E=mu+sqrt(diag(Sigma_C))*z uses coordinate-independent normals. Cross-channel blocks are zero for C, E*, and both Gaussian controls. C and E* generally differ in off-diagonal covariance and are not a matched pair to each other.

## Structured exact action: O(T log T) preprocessing, O(T) sampling/storage

There are at most 2T+1 nonempty intervals and J<=4T+2 nodes. Sort q, precompute L_t=# {q_j<-u_t}, H_t=# {q_j<1-u_t}; equality at the boundary gives the same continuous function value. For arbitrary p define prefix sums P(k)=sum_{j<k}p_j and R(k)=sum_{j<k}q_j p_j. Then

    sum_j p_j g_t(q_j)
      = d[-u_t P(L_t) + R(H_t)-R(L_t)
          + (1-u_t)(P(J)-P(H_t))].

Taking p=w gives mu. Taking p=sqrt(w)*z gives Bz by subtracting mu*P(J). This action is identical to multiplying the finite dense quadrature factor, with only O(J+T) arrays and operations per draw. Sorting/threshold searches are O(T log T) preprocessing per channel; all channels cost O(C T log T) preprocessing and O(C T) storage/draw. No T-by-T array or T-by-J factor is constructed in the shape check. Small-T basis-vector checks compare every implicit factor column with the dense oracle, proving equality of the implemented linear map beyond a sampled covariance check.

For E* a local per-coordinate partition has at most four edges. Local two-node integration gives the same mean and variance with O(T) work/storage. A separate independent polynomial integral checks those scalar moments. No-clip g=d*s*1 has mu=0, Sigma=d^2 a^2/3 * 11^T; use the same quadrature action without saturation.

## Expected affine squared loss (known check)

For fixed affine readout W and squared Euclidean loss, let r=W(vec X+mu)+b-y. Then

    E ||W(vec X+Delta)+b-y||^2 = ||r||^2 + tr(W Sigma W^T).

Both members of each matched pair have exactly this expectation. Small synthetic product quadrature and Gaussian symmetric cubature check the formula independently. This neither applies unchanged to cross-entropy nor asserts equal finite SGD trajectories.

## Prespecified non-Gaussian checks

For u=(0,1), d=1, 0<=a<=1 (epsilon deliberately absent only in this mathematical oracle), mu=(a/4,-a/4), Sigma=(a^2/48)[[5,3],[3,5]]. Its eigenvalues are a^2/6 and a^2/24, both positive when a>0.

Use u=(0,0.2,0.6,1), d=1, a=0.35 for two fixed cubic statistics: (Delta_0-mu_0)^3 and the product of centered coordinates 0,1,2. The first is a^3/32 for C and E*; any matched Gaussian has zero centered cubic. The mixed statistic is integrated exactly for C; E* has zero mixed centered cubic by independence, and every centered Gaussian has zero. These are diagnostics of different distributions, not evidence that those differences improve learning.

## Direction intervention

For the fixed six-token x, a and anchor in protocol.json, subtract the anchor's projections onto 1 and centered mu, then normalize to v. Q=I-2vv^T is symmetric orthogonal, Q1=1 and Qmu=mu. Define Delta'=mu+Q(Delta-mu)=QDelta, while keeping X fixed. Therefore E Delta'=mu, token sum and complete per-draw squared norm are unchanged, and Cov(Delta')=Q Sigma Q^T has the same eigenvalues. This is a valid direction intervention only if ||Q Sigma Q^T-Sigma||_F/||Sigma||_F exceeds the fixed threshold. A vanishing complement or commuting covariance is labeled invalid, never retried with another direction. Shared translation without clipping is unchanged since Q1=1. Rotating the input X or doing a token permutation is not this intervention. The rotated or Gaussian perturbation is not clipped again; bounded support/marginals are not claimed preserved by Q.

## Source scope

Pinned [FAOrCo feature source](https://github.com/learninginvision/FAOrCo-ViT/blob/48424b86061fae71e983d96fc803769185575dbe/models/orco_vit/feature_aug.py) lines 139–159 implements v3; lines 143–157 reduce min/max over tokens per sample/channel, add epsilon, draw a B×1×C uniform tensor, assign clamp, and reconstruct. [Caller](https://github.com/learninginvision/FAOrCo-ViT/blob/48424b86061fae71e983d96fc803769185575dbe/models/orco_vit/Network_vit.py) lines 221–226 leaves CLS separate and augments patches. Those two files are stored as inert .txt, hashed, never imported or executed. The original draws use Torch's default dtype, so our continuous-uniform float64 mathematical oracle is not a bitwise PRNG reproduction.

[FroFA v2 Appendix A2](https://arxiv.org/html/2403.10519v2#A2) shows a clip call without assigning its return. This textual ambiguity prevents claiming the FAOrCo mechanism or this fixture is a bitwise reproduction of FroFA. No conclusion about the paper's actual experiments follows from that snippet.
