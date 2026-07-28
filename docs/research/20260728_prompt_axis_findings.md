# Do better instructions make embedders better at ER blocking?

*2026-07-28. Harness: `examples/research/prompt_axis.py`. Data:
`20260728_prompt_axis_rows.jsonl`; generated tables: `20260728_prompt_axis.md`.*

Modern embedding models are instruction-following — e5, bge, EmbeddingGemma and
Qwen3 each ship a prescribed prompt format, and their published numbers assume
it. langres has a prompt axis in the embedder ladder and asymmetric-prompt
support that PR #242 hardened. This measures whether instructions actually buy
blocking recall, and whether good ones can be *chosen* rather than guessed.

Everything below is **candidate recall at fixed `k`** — never F1 at a threshold,
which would confound the retrieval effect with where the cut falls. Confidence
intervals are paired bootstraps resampled **by gold cluster**.

---

## 1. What each model's own documentation prescribes

Every string below was read from the checkpoint as its author published it — the
model's own `config_sentence_transformers.json` or its model-card `README.md` in
the local Hugging Face snapshot. **None was inferred from another model's
convention**, because these formats have nothing in common and guessing one from
another silently measures the wrong thing.

| model | query side | document side | primary source |
|---|---|---|---|
| `intfloat/e5-base-v2` | `query: ` | `passage: ` | model card `README.md` (snapshot `f52bf8ec…`) L2631: *"Each input text should start with `query: ` or `passage: `"*; FAQ 1 L2687: *"Yes, this is how the model is trained, otherwise you will see a performance degradation."* L2690: `query: `/`passage: ` for **asymmetric** tasks; L2692: `query: ` **on both sides** for **symmetric** tasks. Ships **no** `config_sentence_transformers.json`, so the card is the only source. |
| `BAAI/bge-base-en-v1.5` | `Represent this sentence for searching relevant passages: ` | **none, explicitly** | model card `README.md` (snapshot `a5beb1e3…`) Model List L2679 gives the string verbatim; note [1] L2692: *"In all cases, **no instruction** needs to be added to passages."* L2738-2740 adds that v1.5 was tuned to work **without** it, and L2744: *"The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task."* Its `config_sentence_transformers.json` registers **no prompts at all**. |
| `google/embeddinggemma-300m` | `task: {task description} \| query: ` (default `search result`) | `title: {title \| "none"} \| text: ` | `config_sentence_transformers.json` (snapshot `57c266a7…`) `prompts` map, cross-checked against the card's §Prompt Instructions L344 and task table L366-416. Also registers symmetric templates: `STS`/`PairClassification` → `task: sentence similarity \| query: `, `Clustering` → `task: clustering \| query: `. |
| `Qwen/Qwen3-Embedding-0.6B` | `Instruct: {task description}\nQuery:` | `""` — **literally the empty string** | `config_sentence_transformers.json` (snapshot `c54f2e6e…`): `{"query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", "document": ""}`. Card L129 gives the template, L138 *"No need to add instruction for retrieval documents"*, L54 *"using instructions typically yields an improvement of 1% to 5% … we recommend that developers create tailored instructions specific to their tasks."* |
| `sentence-transformers/all-MiniLM-L6-v2` | — | — | `config_sentence_transformers.json` (snapshot `c9745ed1…`) registers no usable prompts. **Control**: not instruction-trained. |

Note how much they disagree. bge and Qwen3 prescribe a **query-only** recipe —
for them, an unprefixed document side *is* the documented recipe, not a
half-driven mistake. e5 and EmbeddingGemma prescribe **both** sides. e5's
document token is `passage: `; Gemma's is a `title:`/`text:` template. Qwen3's
document prompt is an empty string that would silently look like "no prompt
support" if you only read the config key names.

---

## 2. The arms, and why both halves are always driven

Per model: `none` (baseline), each **documented** recipe, and ER-specific
instructions of our own. Every arm states both halves explicitly.

Our ER instruction frames the task as record identity rather than query→passage
relevance:

> `Find records that describe the same real-world entity as: `

Three arms carry it: `er_symmetric` (both sides), `er_query_only` (query side
only — the **deliberately half-driven trap arm**, measured rather than fallen
into), and for the two models with a real instruction *slot*
(`er_in_official_template`), the ER task description dropped into the
checkpoint's own template shape — exactly what Qwen3's card tells developers to
do.

---

## 3. Evidence the prompts actually reached the encoder

A prompt that never reaches the encoder produces identical numbers, which reads
as *"instructions don't help"*. That bug shipped in this repo once: `search_all()`
served queries from cached corpus vectors, so `query_prompt` was discarded (#239;
#242 turned the remaining cases into a loud `NotImplementedError`). So the
harness **proves** the prompt landed instead of assuming it, on three independent
signals, and **aborts the run** rather than recording a flat row if any fails
(`_prompt_reached_encoder`):

1. **`doc_shift`** — `1 - mean cosine` of the corpus vectors against the
   no-prompt arm. Proves `prompt_name` reached `SentenceTransformer.encode`
   through the **index-build** path.
2. **`query_shift`** — same for the query vectors. Proves `query_prompt` reached
   the encoder through the **search** path — the exact seam that used to discard it.
3. **`doc_query_cosine`** — for a symmetric recipe the two sides must agree
   (cosine `1.0`); for an asymmetric one they must not. This catches a prompt
   that reached one path but was rewritten on the other, which neither shift can
   see alone.

Additionally `pair_jaccard_vs_none` records the Jaccard overlap of the
**candidate-pair set** against the no-prompt arm — proof the changed vectors
changed the *retrieved neighbours*, not merely the geometry.

The signals discriminate exactly as designed. From `all-MiniLM-L6-v2` on
`fodors_zagat`:

| arm | doc shift | query shift | doc·query cos |
|---|---:|---:|---:|
| `none` | 0 | 0 | 1.0000 |
| `er_symmetric` | 0.2486 | 0.2486 | **1.0000** |
| `er_query_only` | **0** | 0.2486 | **0.7514** |

The `er_symmetric` row is the strongest single piece of evidence: the two sides
were driven through *different code paths* — `prompt_name=` on the index build
versus an explicit `prompt=` at search time — and produced vectors that agree to
within `1e-4`. The `er_query_only` row shows the document side correctly left
untouched at exactly `0`, with the query side moved.

Prompt strings also move the vectors by visibly different amounts, which is
itself a sanity check that the *content* is reaching the model: on
`fodors_zagat`, bge's own short instruction shifts vectors by `0.0627` while our
longer ER sentence shifts them by `0.1969`.
