# All incorrect full-query choices: per-fold evidence

Post-result descriptive analysis. Each row compares the actual CV choice to one full-query optimum; all tied optima are retained. CV validation denominator per fold is 20 (B8) or 40 (B16). Query denominator is 1000 per fold. Margins are integer correct-count differences, selected minus reference. These folds are dependent and are not separate statistical trials.

| Encoder/B/episode | Selected → reference | CV selected counts [4 folds] | CV reference counts [4 folds] | CV margin | Fold-query margins [4 folds] | Full-query regret pp |
|:---|:---|:---|:---|---:|:---|---:|
| dinov2_small/8/1 | weak → clean | [17, 19, 15, 19] | [18, 18, 13, 17] | 4 | [-4, 13, -5, 1] | 1.0 |
| dinov2_small/8/2 | clean → weak | [18, 17, 15, 16] | [17, 17, 15, 16] | 1 | [-9, -65, -39, -34] | 4.4 |
| dinov2_small/8/4 | weak → ra | [19, 17, 18, 17] | [19, 19, 19, 14] | 0 | [-5, -13, -6, 11] | 0.8 |
| dinov2_small/8/9 | ra → weak | [17, 15, 16, 19] | [17, 14, 16, 18] | 2 | [-6, -9, -40, -30] | 2.9 |
| dinov2_small/16/2 | ra → weak | [34, 36, 35, 35] | [33, 34, 33, 36] | 4 | [-9, 1, -3, -13] | 1.3 |
| dinov2_small/16/4 | ra → weak | [36, 33, 35, 36] | [37, 31, 35, 35] | 2 | [-9, -6, -4, 0] | 0.3 |
| dinov2_small/16/6 | ra → weak | [36, 34, 35, 36] | [36, 33, 33, 37] | 2 | [-13, -3, -8, -13] | 1.6 |
| dinov2_small/16/8 | ra → weak | [37, 34, 33, 33] | [37, 31, 32, 33] | 4 | [-27, -18, -33, -13] | 1.9 |
| deit_tiny/8/2 | ra → weak | [15, 13, 15, 16] | [16, 15, 12, 15] | 1 | [-26, -32, 2, -2] | 1.7 |
| deit_tiny/8/4 | clean → weak | [17, 16, 15, 12] | [16, 14, 15, 13] | 2 | [-35, -43, -34, -36] | 4.0 |
| deit_tiny/8/9 | weak → ra | [17, 16, 17, 20] | [18, 17, 15, 17] | 3 | [0, 9, 11, -9] | 1.1 |
| deit_tiny/8/11 | ra → weak | [12, 13, 16, 16] | [13, 14, 15, 13] | 2 | [-11, -21, -18, 7] | 1.3 |
| deit_tiny/16/0 | weak → ra | [33, 34, 34, 30] | [33, 31, 33, 32] | 2 | [3, 3, 4, -2] | 0.3 |
| deit_tiny/16/1 | weak → ra | [32, 29, 32, 37] | [32, 30, 31, 34] | 3 | [-12, -11, -9, 15] | 1.8 |
| deit_tiny/16/4 | clean → ra | [35, 33, 31, 31] | [30, 30, 32, 33] | 5 | [-39, -26, -37, -42] | 3.3 |
| deit_tiny/16/5 | ra → weak | [36, 35, 29, 36] | [34, 34, 29, 34] | 5 | [19, 8, -29, 3] | 0.5 |
| deit_tiny/16/6 | clean → ra | [34, 34, 35, 31] | [33, 35, 33, 31] | 2 | [-47, -57, -39, -55] | 4.2 |
| deit_tiny/16/8 | weak → ra | [32, 31, 31, 32] | [30, 30, 33, 31] | 2 | [-1, 1, 4, -7] | 1.4 |
| deit_tiny/16/9 | clean → weak | [33, 35, 37, 34] | [36, 33, 36, 34] | 0 | [-39, -11, -32, -38] | 3.4 |
| deit_tiny/16/9 | clean → ra | [33, 35, 37, 34] | [34, 33, 35, 35] | 2 | [-43, -9, -37, -53] | 3.4 |
