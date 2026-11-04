# All prespecified support settings

Accuracy and differences are percentage points. No setting omitted. C=clean, W=weak RRC+flip, R=weak+RandAugment. Query size=1000, fixed and exploratory.

## B=8 labels/class

| Episode | C | W | R | CV choice | Selected−W | Selected−R | Oracle regret | CV best / fold-query best / full-query best |
|---:|---:|---:|---:|:---:|---:|---:|---:|:---|
| 0 | 81.6 | 82.4 | 78.2 | weak | +0.0 | +4.2 | 0.0 | weak / weak / weak |
| 1 | 81.1 | 80.1 | 80.3 | weak | +0.0 | -0.2 | 1.0 | weak / ra / clean |
| 2 | 80.7 | 85.1 | 84.0 | clean | -4.4 | -3.3 | 4.4 | clean / weak / weak |
| 3 | 80.6 | 82.9 | 80.0 | weak | +0.0 | +2.9 | 0.0 | weak / weak / weak |
| 4 | 79.9 | 81.9 | 82.7 | weak | +0.0 | -0.8 | 0.8 | weak / ra / ra |
| 5 | 80.1 | 79.8 | 82.9 | ra | +3.1 | +0.0 | 0.0 | ra / ra / ra |
| 6 | 77.2 | 81.1 | 81.9 | ra | +0.8 | +0.0 | 0.0 | ra / ra / ra |
| 7 | 80.3 | 81.9 | 80.7 | weak | +0.0 | +1.2 | 0.0 | weak / weak / weak |
| 8 | 82.3 | 83.3 | 81.2 | weak | +0.0 | +2.1 | 0.0 | weak / weak / weak |
| 9 | 83.8 | 84.8 | 81.9 | ra | -2.9 | +0.0 | 2.9 | ra / weak / weak |
| 10 | 81.2 | 81.8 | 83.3 | ra | +1.5 | +0.0 | 0.0 | ra / ra / ra |
| 11 | 82.6 | 84.1 | 83.7 | weak | +0.0 | +0.4 | 0.0 | weak / weak / weak |

## B=16 labels/class

| Episode | C | W | R | CV choice | Selected−W | Selected−R | Oracle regret | CV best / fold-query best / full-query best |
|---:|---:|---:|---:|:---:|---:|---:|---:|:---|
| 0 | 87.7 | 88.7 | 86.6 | weak | +0.0 | +2.1 | 0.0 | weak / weak / weak |
| 1 | 87.6 | 87.6 | 85.2 | weak | +0.0 | +2.4 | 0.0 | weak / weak / clean |
| 2 | 87.9 | 88.8 | 87.5 | ra | -1.3 | +0.0 | 1.3 | ra / weak / weak |
| 3 | 87.8 | 88.8 | 86.6 | weak | +0.0 | +2.2 | 0.0 | weak / weak / weak |
| 4 | 88.1 | 88.7 | 88.4 | ra | -0.3 | +0.0 | 0.3 | ra / weak / weak |
| 5 | 83.7 | 84.4 | 85.8 | ra | +1.4 | +0.0 | 0.0 | ra / ra / ra |
| 6 | 85.8 | 88.0 | 86.4 | ra | -1.6 | +0.0 | 1.6 | ra / weak / weak |
| 7 | 86.4 | 86.8 | 85.7 | weak | +0.0 | +1.1 | 0.0 | weak / weak / weak |
| 8 | 85.4 | 88.1 | 86.2 | ra | -1.9 | +0.0 | 1.9 | ra / weak / weak |
| 9 | 87.5 | 87.5 | 87.8 | ra | +0.3 | +0.0 | 0.0 | ra / weak / ra |
| 10 | 86.4 | 88.3 | 86.8 | weak | +0.0 | +1.5 | 0.0 | weak / weak / weak |
| 11 | 87.6 | 88.3 | 87.0 | weak | +0.0 | +1.3 | 0.0 | weak / weak / weak |

Policy-label changes under ties can occur without an accuracy loss; the JSON/CSV reports strict losses separately. Oracle uses query labels after all deployed selections were frozen.
