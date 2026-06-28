# Bootstrap Report

## Blocking (pair-completeness)
- Pair-completeness (candidate recall): 0.9911
- Candidate precision: 0.0803
- Total candidates: 1382
- Missed matches: 1

## Teacher-vs-truth agreement
- Evaluated pairs: 213
- Accuracy: 0.8498
- Precision: 0.7801
- Recall: 0.9910
- F1: 0.8730
- Cohen's kappa: 0.6954
- MCC: 0.7256

## Calibration (P(match) score vs. is-match)
- Evaluated pairs: 213
- Brier score (primary): 0.1469
- ECE (equal-mass, 8 bins): 0.1363
- Reliability bins (mean_conf -> observed_freq, count):
  - 0.0000 -> 0.0000 (n=67)
  - 0.0500 -> 0.0000 (n=3)
  - 0.8180 -> 0.7333 (n=15)
  - 0.9882 -> 0.9697 (n=33)
  - 1.0000 -> 0.7158 (n=95)

## Routing / coverage
- Total candidates: 1382
- Mined: 1382
- Labeled: 1382
- Skipped: 0
- With ground truth: 213
- Total cost (USD): 1.2848

## Agreement convergence
- Points: 213
- Final F1 @ 213 labels: 0.8730
