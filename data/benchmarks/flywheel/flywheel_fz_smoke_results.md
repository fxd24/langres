# Flywheel closed loop -- Fodors-Zagat wiring smoke (T8a)

- Dataset: **fodors-zagat**
- Teacher model: `openrouter/openai/gpt-4o-mini`
- Mode: **REAL (paid)**
- Spend cap (budget_usd): **$2.00**
- Teacher spend: **$0.0658** (REAL)

## Economics

| metric | value |
| --- | --- |
| escalation rate | 31.7% (19/60 held-out pairs) |
| frontier-call reduction | 68.3% |
| escalated-pair accuracy | 89.5% |
| audit-slice disagreement | 6.7% |

## Pairwise F1 on the held-out split (one threshold cuts every stream)

| judge | F1 | precision | recall |
| --- | --- | --- | --- |
| teacher (frontier) | 0.944 | 1.000 | 0.895 |
| student (cheap) | 1.000 | 1.000 | 1.000 |
| cascade | 0.944 | 1.000 | 0.895 |

## Notes

WIRING SMOKE, not an economics claim: Fodors-Zagat is easy, so the cheap student already resolves the held-out split and the cascade's value here is the frontier-call reduction, not an F1 gain. Real teacher/student economics live in flywheel_amazon_google.py.
