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

The guard has been **seen to fail**, which is the only reason to trust it. It
aborted the first Qwen3 run at `cosine=1.000126` on the no-prompt arm. A cosine
*above* 1 is arithmetically impossible for two distinct unit vectors, so that
value was `float16` accumulation error rather than divergent encode paths; the
tolerance is now bounded on both sides by measured values and
`tests/examples/test_prompt_axis.py` exercises every failure branch plus the
margin, so the gate cannot be quietly loosened into uselessness.

---

## 4. Results

Full per-model × per-benchmark tables with all four `k` values are in
`20260728_prompt_axis.md`. Everything below is `k=20`. **Nothing is averaged
across models** — the whole question was whether the effect is model-specific,
and it emphatically is.

`fodors_zagat` is **saturated**: every model reaches recall `1.0000` at `k=20`
with no prompt at all, so it can only detect harm, never benefit. It is a
control, not evidence. (It did detect harm once — see Gemma's `clustering` arm.)

### 4.1 The headline: the checkpoint's own *retrieval* prefix is the lever

**`google/embeddinggemma-300m`** — the model David's hypothesis was about, and
the cleanest result in the sweep. Its documented Retrieval pair
(`title: none | text: ` on documents, `task: search result | query: ` on queries)
is the **only arm positive and clear of zero on all three unsaturated
benchmarks**, and it is never negative anywhere:

| arm | kind | abt_buy | amazon_google | wdc_computers |
|---|---|---|---|---|
| `official_retrieval` | documented | **+0.0088** [+0.0030, +0.0147] | **+0.0371** [+0.0264, +0.0478] | **+0.0394** [+0.0245, +0.0535] |
| `official_sts` | documented | −0.0127 [−0.0226, −0.0029] | +0.0224 [+0.0132, +0.0328] | −0.0410 [−0.0587, −0.0243] |
| `official_clustering` | documented | −0.0721 [−0.0869, −0.0559] | −0.0164 [−0.0272, −0.0056] | **−0.3183** [−0.3500, −0.2893] |
| `er_in_official_template` | ours | −0.0181 [−0.0275, −0.0088] | +0.0238 [+0.0136, +0.0346] | **−0.1589** [−0.1847, −0.1358] |
| `er_symmetric` | ours | −0.0078 [−0.0147, −0.0020] | +0.0118 [+0.0043, +0.0198] | +0.0004 [−0.0139, +0.0139] |
| `er_query_only` | trap | +0.0010 [−0.0039, +0.0059] | +0.0136 [+0.0043, +0.0244] | +0.0172 [+0.0054, +0.0290] |

So **yes — instructions measurably help an instruction-following model**, and the
gain is worth having: `wdc_computers` recall goes `0.7786 → 0.8128`.

But note the second row set. **"Documented" is not sufficient.** Gemma's
`clustering` template — equally official, equally published — costs **−0.3183
recall** on `wdc_computers`, and is the single arm anywhere in this sweep that
broke the saturated `fodors_zagat` control (`1.0000 → 0.9375`). Picking the
wrong official template is far more damaging than using no prompt at all.

**`intfloat/e5-base-v2`** — same shape, smaller magnitude. The card's
**asymmetric** retrieval recipe wins; its **symmetric** rule of thumb loses:

| arm | abt_buy | amazon_google | wdc_computers |
|---|---|---|---|
| `official_asymmetric` (`query:`/`passage:`) | **+0.0069** [+0.0020, +0.0128] | +0.0020 [−0.0010, +0.0057] | **+0.0152** [+0.0009, +0.0308] |
| `official_symmetric` (`query:` both sides) | −0.0020 [−0.0059, +0.0020] | **−0.0029** [−0.0059, −0.0003] | **−0.0145** [−0.0288, −0.0023] |

This **contradicts e5's own model card**, and is the most useful negative result
here. The card's FAQ says to use `query: ` on both sides for *"symmetric tasks
such as semantic similarity, paraphrase retrieval"*. ER blocking over a pooled
corpus looks exactly like that description — and does not behave like it. Reading
the card was necessary but not sufficient; only measurement settled it.

**`BAAI/bge-base-en-v1.5`** — the largest single gain in the sweep, from a
query-only recipe that is documented as query-only:

| arm | abt_buy | amazon_google | wdc_computers |
|---|---|---|---|
| `official_query_instruction` | **+0.0093** [+0.0044, +0.0152] | +0.0021 [−0.0014, +0.0061] | **+0.1224** [+0.1013, +0.1451] |
| `official_symmetric` (same string, both sides) | **+0.0064** [+0.0015, +0.0118] | +0.0002 [−0.0035, +0.0037] | **+0.1084** [+0.0877, +0.1299] |

`wdc_computers` recall goes `0.6103 → 0.7201`. Note that bge's instruction helps
in *either* placement — query-only (documented) or both-sides (not documented) —
which is the tell that what matters is the **string being one the checkpoint was
trained on**, not where it sits.

### 4.2 The counter-headline: writing a better English instruction does not work

This answers the underlying question better than the positive result does. Our
ER instruction — *"Find records that describe the same real-world entity as: "* —
is a genuinely better *description* of blocking than "retrieve relevant passages
that answer the query". It describes the task correctly. It is also, on the
models where it matters most, **actively harmful**:

| model | benchmark | `er_symmetric` Δ recall | 95% CI |
|---|---|---:|---|
| `intfloat/e5-base-v2` | wdc_computers | **−0.1038** | [−0.1242, −0.0840] |
| `BAAI/bge-base-en-v1.5` | wdc_computers | **−0.0775** | [−0.0988, −0.0556] |
| `all-MiniLM-L6-v2` (control) | abt_buy | **−0.0667** | [−0.0820, −0.0500] |
| `all-MiniLM-L6-v2` (control) | wdc_computers | **−0.0632** | [−0.0843, −0.0441] |
| `intfloat/e5-base-v2` | abt_buy | **−0.0436** | [−0.0568, −0.0314] |
| `google/embeddinggemma-300m` | wdc_computers | +0.0004 | [−0.0139, +0.0139] |

And dropping our ER task description into Gemma's *own template shape*
(`task: entity resolution | query: `) — precisely what Qwen3's card instructs
developers to do — costs **−0.1589** [−0.1847, −0.1358] on `wdc_computers`.

**The lever is the checkpoint's trained prefix, not better English.** A prompt to
these models is not an instruction that is understood; it is a token sequence
whose embedding geometry was fixed during contrastive training. A prefix the
model never saw in training moves every vector by a large, roughly common
displacement (our ER sentence shifts vectors by `0.20`–`0.38`, versus `0.02`–`0.10`
for the models' own prefixes) and that displacement crowds out the
record-specific signal blocking depends on.

### 4.3 The control behaved as a control should

`sentence-transformers/all-MiniLM-L6-v2` is not instruction-trained and registers
no prompts. **No arm helped it on any benchmark.** `er_symmetric` significantly
hurt on 2 of 3, `er_query_only` significantly hurt on 1 of 3, and nothing was
significantly positive anywhere. This is the null result the design predicted,
and its absence would have invalidated the whole sweep.

### 4.4 The half-driven trap, measured

`er_query_only` drives the query side and leaves documents bare — the state #242
warns about. Measured, it is **not uniformly worse**, which is worth saying
plainly rather than repeating the folk rule:

- On the models with no matching trained prefix it is mildly negative
  (e5 `−0.0020` to `−0.0045`; MiniLM `−0.0029` to `−0.0122`).
- On Gemma it is mildly *positive* (`+0.0136` to `+0.0172`).
- For **bge and Qwen3 the query-only shape is the documented recipe**, and for bge
  it produces the largest gain in the sweep.

So "always drive both halves" is the wrong rule. The right rule is **drive both
halves the way the checkpoint was trained**, which for two of these four models
means deliberately leaving the document side bare.
