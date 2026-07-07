# Phase 1 -- LLMJudge honest full-split placement (paid)

Prompted `LLMJudge` on the **full standard test split**, threshold derived honestly (on VALID, or a fixed constant) and applied to all of test. `argmax-F1` is the leaky ceiling (cut tuned on test) shown only for the honesty delta. `real cost` is OpenRouter's actual billed spend. The gap columns place the honest F1 against the $0 RandomForestJudge floor (0.360 AG / 0.404 Abt-Buy) and the Ditto SOTA band (0.756 / 0.893).

| model | dataset | honest F1 | honest P | honest R | threshold | argmax-F1 | real cost USD | provider | gap to RF floor | gap to Ditto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openrouter/deepseek/deepseek-v4-flash` | abt_buy | 0.6803 | 0.5236 | 0.9709 | 0.5000 | 0.6866 | $0.1839 | AkashML, Alibaba, AtlasCloud, Baidu, DeepInfra, DigitalOcean, Fireworks, GMICloud, Morph, Novita, Parasail, SiliconFlow, StreamLake, Venice, WandB | +0.2766 | +0.2127 |
| `openrouter/deepseek/deepseek-v4-flash` | amazon_google | 0.5748 | 0.4165 | 0.9274 | 0.5000 | 0.5902 | $0.1892 | AkashML, Alibaba, AtlasCloud, Baidu, DeepInfra, DigitalOcean, Fireworks, GMICloud, Morph, Novita, Parasail, SiliconFlow, StreamLake, Venice, WandB | +0.2152 | +0.1812 |
| `openrouter/deepseek/deepseek-v4-pro` | abt_buy | 0.7366 | 0.5935 | 0.9709 | 0.5000 | 0.7407 | $2.2711 | Alibaba, AtlasCloud, Baidu, BaseTen, DeepInfra, DigitalOcean, GMICloud, Novita, Parasail, SiliconFlow, StreamLake, Together, Venice, WandB | +0.3330 | +0.1564 |
| `openrouter/deepseek/deepseek-v4-pro` | amazon_google | 0.6141 | 0.4580 | 0.9316 | 0.5000 | 0.6250 | $2.5568 | Alibaba, AtlasCloud, Baidu, BaseTen, DeepInfra, DigitalOcean, GMICloud, Novita, Parasail, SiliconFlow, StreamLake, Together, Venice, WandB | +0.2545 | +0.1419 |

## Reading

- **honest F1** is the placement number: no test-label peeking.
- **gap to RF floor** > 0 means the paid LLM judge beats the $0 local baseline; < 0 means the thin single-metric floor is (surprisingly) ahead.
- **gap to Ditto** is the distance still open to the SOTA band.
- **real cost USD** is the actual OpenRouter spend for that cell (per-`(model, dataset)` JSON has the served-provider breakdown).
