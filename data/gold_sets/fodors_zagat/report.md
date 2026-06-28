# Bootstrap Report

## Blocking (pair-completeness)
- Pair-completeness (candidate recall): 0.9911
- Candidate precision: 0.0803
- Total candidates: 1382
- Missed matches: 1

## Teacher-vs-truth agreement
- Evaluated pairs: 1382
- Accuracy: 0.8025
- Precision: 0.2880
- Recall: 0.9910
- F1: 0.4462
- Cohen's kappa: 0.3675
- MCC: 0.4722

## Calibration (P(match) score vs. is-match)
- Evaluated pairs: 1382
- Brier score (primary): 0.1941
- ECE (equal-mass, 8 bins): 0.1948
- Reliability bins (mean_conf -> observed_freq, count):
  - 0.0000 -> 0.0000 (n=982)
  - 0.0100 -> 0.0000 (n=1)
  - 0.0500 -> 0.0000 (n=10)
  - 0.1000 -> 0.1667 (n=6)
  - 0.1500 -> 0.0000 (n=1)
  - 0.9022 -> 0.4444 (n=36)
  - 0.9992 -> 0.2717 (n=346)

## Routing / coverage
- Total candidates: 1382
- Mined: 1382
- Labeled: 1382
- Skipped: 0
- With ground truth: 1382
- Total cost (USD): 1.2848

## Agreement convergence
- Points: 1382
- Final F1 @ 1382 labels: 0.4462
