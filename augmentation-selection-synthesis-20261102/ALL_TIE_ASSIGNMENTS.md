# Complete CV-tie candidate sensitivity

Original tie rule remains clean > weak > RA. All query-labelled choices below are diagnostic bounds, not a replacement policy. No candidates or assignments are omitted.

## dinov2_small B=8, episode 4

Original choice: weak. Tied OOF correct: 71/80.

| Candidate | CV correct by fold | Fold-query mean % | Full-query % | Regret pp |
|:---|:---|---:|---:|---:|
| weak | [19, 17, 18, 17] | 80.400 | 81.9 | 0.8 |
| ra | [19, 19, 19, 14] | 80.725 | 82.7 | 0.0 |

## dinov2_small B=8, episode 7

Original choice: weak. Tied OOF correct: 64/80.

| Candidate | CV correct by fold | Fold-query mean % | Full-query % | Regret pp |
|:---|:---|---:|---:|---:|
| weak | [14, 15, 17, 18] | 80.300 | 81.9 | 0.0 |
| ra | [14, 16, 16, 18] | 78.750 | 80.7 | 1.2 |

## deit_tiny B=16, episode 6

Original choice: clean. Tied OOF correct: 134/160.

| Candidate | CV correct by fold | Fold-query mean % | Full-query % | Regret pp |
|:---|:---|---:|---:|---:|
| clean | [34, 34, 35, 31] | 76.225 | 79.0 | 4.2 |
| weak | [35, 34, 33, 32] | 80.600 | 82.8 | 0.4 |

## deit_tiny B=16, episode 9

Original choice: clean. Tied OOF correct: 139/160.

| Candidate | CV correct by fold | Fold-query mean % | Full-query % | Regret pp |
|:---|:---|---:|---:|---:|
| clean | [33, 35, 37, 34] | 76.825 | 78.5 | 3.4 |
| weak | [36, 33, 36, 34] | 79.825 | 81.9 | 0.0 |

## All assignments: dinov2_small:B8

Variable episodes: [4, 7]. Other choices remain exactly the original unique CV best.

| Choices at variable episodes | Mean accuracy | CV−weak | CV−RA | Mean regret | C/W/R selection counts |
|:---|---:|---:|---:|---:|:---|
| ['weak', 'weak'] | 82.275 | -0.158 | +0.542 | 0.758 | [1, 7, 4] |
| ['weak', 'ra'] | 82.175 | -0.258 | +0.442 | 0.858 | [1, 6, 5] |
| ['ra', 'weak'] | 82.342 | -0.092 | +0.608 | 0.692 | [1, 6, 5] |
| ['ra', 'ra'] | 82.242 | -0.192 | +0.508 | 0.792 | [1, 5, 6] |

Marginal frequency bounds (not all jointly realizable): {'clean': [1, 1], 'weak': [5, 7], 'ra': [4, 6]}

## All assignments: dinov2_small:B16

Variable episodes: []. Other choices remain exactly the original unique CV best.

| Choices at variable episodes | Mean accuracy | CV−weak | CV−RA | Mean regret | C/W/R selection counts |
|:---|---:|---:|---:|---:|:---|
| [] | 87.550 | -0.283 | +0.883 | 0.425 | [0, 6, 6] |

Marginal frequency bounds (not all jointly realizable): {'clean': [0, 0], 'weak': [6, 6], 'ra': [6, 6]}

## All assignments: deit_tiny:B8

Variable episodes: []. Other choices remain exactly the original unique CV best.

| Choices at variable episodes | Mean accuracy | CV−weak | CV−RA | Mean regret | C/W/R selection counts |
|:---|---:|---:|---:|---:|:---|
| [] | 76.133 | -0.267 | -0.342 | 0.675 | [1, 4, 7] |

Marginal frequency bounds (not all jointly realizable): {'clean': [1, 1], 'weak': [4, 4], 'ra': [7, 7]}

## All assignments: deit_tiny:B16

Variable episodes: [6, 9]. Other choices remain exactly the original unique CV best.

| Choices at variable episodes | Mean accuracy | CV−weak | CV−RA | Mean regret | C/W/R selection counts |
|:---|---:|---:|---:|---:|:---|
| ['clean', 'clean'] | 81.500 | -0.550 | -1.150 | 1.242 | [3, 5, 4] |
| ['clean', 'weak'] | 81.783 | -0.267 | -0.867 | 0.958 | [2, 6, 4] |
| ['weak', 'clean'] | 81.817 | -0.233 | -0.833 | 0.925 | [2, 6, 4] |
| ['weak', 'weak'] | 82.100 | +0.050 | -0.550 | 0.642 | [1, 7, 4] |

Marginal frequency bounds (not all jointly realizable): {'clean': [1, 3], 'weak': [5, 7], 'ra': [4, 4]}

