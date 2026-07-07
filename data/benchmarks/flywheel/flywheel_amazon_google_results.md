# Flywheel closed loop -- Amazon-Google economics (T8b)

- Dataset: **amazon-google**
- Teacher model: `openrouter/openai/gpt-4o-mini`
- Mode: **REAL (paid)**
- Spend cap (budget_usd): **$8.00**
- Teacher spend: **$1.0457** (REAL)

## Economics

| metric | value |
| --- | --- |
| escalation rate | 22.5% (225/1000 held-out pairs) |
| frontier-call reduction | 77.5% |
| escalated-pair accuracy | 84.9% |
| audit-slice disagreement | 20.0% |

## Pairwise F1 on the held-out split (one threshold cuts every stream)

| judge | F1 | precision | recall |
| --- | --- | --- | --- |
| teacher (frontier) | 0.557 | 0.407 | 0.885 |
| student (cheap) | 0.302 | 0.228 | 0.448 |
| cascade | 0.381 | 0.283 | 0.583 |

## Notes

Economics on a bounded Amazon-Google subset (2000 candidate pairs, seed 7). AG products mapped onto FZRecord as name=title, addr=manufacturer (cosmetic field labels; values drive the result). The positive rate is kept near the natural ~10% with a floor so both split halves span both label classes. Honest reading -- the COST lever is unambiguous: 77.5% fewer frontier calls for $1.05, at 84.9% escalated-verdict accuracy. QUALITY is the hard-data reality: AG is hard, so the *uncompiled* frontier teacher is itself low-precision (0.407), so its silver labels are noisy; a student trained on only 30 human corrections + that noisy silver is weak (F1 0.302). The cascade recovers part of the gap (F1 0.381 > the cheap student's 0.302) but stays below the teacher (F1 0.557) -- a cascade cannot exceed the signal it distils. The lever that closes the gap is more human labels at the margin, which is exactly what the loop accumulates. These are the real-model economics behind GETTING_STARTED's cascade story; the FZ/simulated runs are wiring/plumbing only.
