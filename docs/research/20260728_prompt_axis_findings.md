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

> **Two estimators, and the Δ is not the difference of the recalls.** A `recall`
> figure is *micro* — the fraction of all gold pairs captured. Every **Δ** and
> **CI** below is *macro over records* — the mean per-record fraction of that
> record's gold partners captured — because a paired bootstrap needs a per-entity
> score to resample by cluster, and a corpus-wide micro ratio has no per-entity
> decomposition. The two diverge when clusters differ in size: for
> bge/`wdc_computers`, micro recall moves `+0.1098` where the macro Δ reads
> `+0.1224`. Both are honest; they answer slightly different questions, and the
> interval belongs to the macro one. (Distinction surfaced by automated review on
> PR #252; the column is now named `delta_per_record_recall` in the rows.)

---

## The answer, in four lines

1. **Yes, instructions help — but only the checkpoint's own *retrieval* prefix.**
   EmbeddingGemma's documented Retrieval pair is the only **documented** arm in
   the sweep that is positive and clear of zero on all three unsaturated
   benchmarks (`wdc_computers` recall `0.7786 → 0.8128`). bge's documented query
   instruction is worth **+0.1224** recall on the same benchmark. (Exactly one
   *other* arm clears zero on all three — Qwen3 with **our** text in its
   template — which is the whole subject of line 3.)
2. **"Documented" is NOT sufficient — this is the trap.** Gemma's `clustering`
   template is every bit as official as its retrieval one and costs **−0.3183**
   recall on `wdc_computers`. Anyone skimming for "use the official prompt" will
   take away the wrong instruction. You must use the checkpoint's *retrieval*
   template, and you must verify it on your data.
3. **Writing your own instruction is not the lever.** A raw ER sentence used as a
   bare prefix is reliably harmful (up to **−0.1038**). Putting our ER task text
   inside the checkpoint's own template is a **coin flip**: it helped Qwen3
   (+0.0388) and cost Gemma **−0.1589** — and it never beat the model's own
   default *on candidate recall*, even when that default describes *web search*.
   Take the checkpoint's
   retrieval prompt; do not author a better one. (§4.2 has the three tiers.)
4. **The effect is strongly model-specific**, which is why nothing here is
   averaged. On `wdc_computers`, the *identical* `er_in_official_template` arm is
   **−0.1589** for EmbeddingGemma and **+0.0388** for Qwen3 — a spread of
   **0.1977** across two models on one benchmark with one prompt. A
   non-instruction-trained control gained nothing anywhere.

**The operating rule: drive the sides the checkpoint was trained on.** *Not*
"always drive both halves" — that sounds sensible and is wrong. For bge and Qwen3
the documented recipe is **query-side only, document side deliberately bare**, and
for bge that one-sided recipe produces the largest gain measured here. §4.4 has
the numbers.

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

### A note on the saturated benchmark, because it earned its keep

`fodors_zagat` is **saturated**: every model reaches recall `1.0000` at `k=20`
with no prompt at all, so it can only ever detect *harm*, never benefit. The
usual move is to drop such a benchmark from a portfolio as signal-free.

That would have been a mistake here. Of the 25 arms measured against it, exactly
one moved it — Gemma's documented `clustering` template, `1.0000 → 0.9375`
[−0.1074, −0.0179] — and that is the same arm that turned out to be catastrophic
(**−0.3183**) on `wdc_computers`. The saturated benchmark acted as a clean
**harm detector**: silence from it means "this arm is not catastrophic", and its
one non-zero reading flagged the single worst configuration in the sweep.

Generalisable: keep saturated benchmarks in a portfolio as harm detectors. They
carry no ranking information, but a regression that breaks a benchmark nothing
else could break is worth catching, and their silence is cheap.

### 4.1 The headline: the checkpoint's own *retrieval* prefix is the lever

**`google/embeddinggemma-300m`** — the model David's hypothesis was about, and
the cleanest result in the sweep. Its documented Retrieval pair
(`title: none | text: ` on documents, `task: search result | query: ` on queries)
is the **only documented arm positive and clear of zero on all three unsaturated
benchmarks**, and it is never negative anywhere. (One non-documented arm also
manages it — Qwen3 running *our* ER text inside its own template, §4.1 below —
so the uniqueness is among the published recipes, not among all 25 arms.)

| arm | kind | abt_buy | amazon_google | wdc_computers |
|---|---|---|---|---|
| `official_retrieval` | documented | **+0.0088** [+0.0039, +0.0157] | **+0.0371** [+0.0271, +0.0482] | **+0.0394** [+0.0244, +0.0551] |
| `official_sts` | documented | −0.0127 [−0.0216, −0.0039] | +0.0224 [+0.0130, +0.0326] | −0.0410 [−0.0584, −0.0249] |
| `official_clustering` | documented | −0.0721 [−0.0874, −0.0564] | −0.0164 [−0.0279, −0.0053] | **−0.3183** [−0.3492, −0.2899] |
| `er_in_official_template` | ours | −0.0181 [−0.0279, −0.0079] | +0.0238 [+0.0145, +0.0338] | **−0.1589** [−0.1821, −0.1361] |
| `er_symmetric` | ours | −0.0078 [−0.0138, −0.0020] | +0.0118 [+0.0046, +0.0196] | +0.0004 [−0.0131, +0.0149] |
| `er_query_only` | trap | +0.0010 [−0.0039, +0.0059] | +0.0136 [+0.0031, +0.0233] | +0.0172 [+0.0064, +0.0300] |

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
| `official_asymmetric` (`query:`/`passage:`) | **+0.0069** [+0.0010, +0.0128] | +0.0020 [−0.0012, +0.0056] | +0.0152 [+0.0000, +0.0310] |
| `official_symmetric` (`query:` both sides) | −0.0020 [−0.0059, +0.0020] | **−0.0029** [−0.0059, −0.0004] | **−0.0145** [−0.0274, −0.0023] |

> **The `wdc_computers` cell is a boundary case, not a win.** `+0.0152` looks
> like the largest e5 effect in the table, but its interval closes **exactly on
> zero** — it does not exclude it, so this benchmark does not by itself establish
> that the asymmetric recipe helps e5. The claim rests on `abt_buy`
> (`+0.0069 [+0.0010, +0.0128]`), which does. An earlier draft of this doc read
> `[+0.0009, +0.0308]` here and called it significant; that interval came from a
> run whose bootstrap was not reproducible (gold clusters arrive as `set`s, so
> per-process hash randomisation changed the resampling order despite the fixed
> seed). After the fix, every point estimate in this document is unchanged and
> this is the **only** verdict that moved.

This **contradicts e5's own model card**, and is the most useful negative result
here. The card's FAQ says to use `query: ` on both sides for *"symmetric tasks
such as semantic similarity, paraphrase retrieval"*. ER blocking over a pooled
corpus looks exactly like that description — and does not behave like it. Reading
the card was necessary but not sufficient; only measurement settled it. Note the
asymmetric recipe's advantage over the symmetric one is clear on `wdc_computers`
even though its own margin over *no prompt* is not: `−0.0145` and `+0.0152` have
non-overlapping intervals.

**`BAAI/bge-base-en-v1.5`** — the largest single gain in the sweep, from a
query-only recipe that is documented as query-only:

| arm | abt_buy | amazon_google | wdc_computers |
|---|---|---|---|
| `official_query_instruction` | **+0.0093** [+0.0039, +0.0157] | +0.0021 [−0.0016, +0.0062] | **+0.1224** [+0.1006, +0.1449] |
| `official_symmetric` (same string, both sides) | **+0.0064** [+0.0010, +0.0122] | +0.0002 [−0.0037, +0.0042] | **+0.1084** [+0.0868, +0.1302] |

`wdc_computers` recall goes `0.6103 → 0.7201`. Note that bge's instruction helps
in *either* placement — query-only (documented) or both-sides (not documented) —
which is the tell that what matters is the **string being one the checkpoint was
trained on**, not where it sits.

**`Qwen/Qwen3-Embedding-0.6B`** — the only model where *our own* task text also
helped, and the reason the story below is three tiers rather than two:

| arm | kind | abt_buy | amazon_google | wdc_computers |
|---|---|---|---|---|
| `official_query_instruct` | documented | **+0.0088** [+0.0029, +0.0147] | +0.0040 [−0.0002, +0.0084] | **+0.0646** [+0.0469, +0.0829] |
| `er_in_official_template` | ours | **+0.0088** [+0.0029, +0.0147] | **+0.0044** [+0.0012, +0.0088] | **+0.0388** [+0.0237, +0.0551] |
| `er_symmetric` | ours | +0.0010 [−0.0020, +0.0040] | +0.0005 [−0.0049, +0.0057] | **−0.0907** [−0.1150, −0.0660] |

Qwen3's card claims instructions are worth *"an improvement of 1% to 5%"*. We
measured `+0.4%` to `+6.5%` — consistent with the claim, and the only model card
prediction in this sweep that survived contact with the data.

But note what beat what. The card also says to *"create tailored instructions
specific to your tasks"*. We did exactly that (`er_in_official_template`), and it
**lost to the model's own generic default** on `wdc_computers` (+0.0388 vs
+0.0646) — despite that default being an instruction about retrieving **web
search passages**, a task description that is simply wrong for entity resolution.

The `amazon_google` column is the one place our text *looks* better: its macro Δ
clears zero (`+0.0044`) where the default's does not (`+0.0040`). Do not read that
as a win. The two intervals overlap almost completely, and on **candidate recall**
— the quantity being maximised — our arm is actually *lower* (`0.8367` vs
`0.8374`). Across all three benchmarks our text never achieves higher recall than
the checkpoint's own default; it ties on `abt_buy` and is clearly worse on
`wdc_computers`.

### 4.2 The counter-headline: writing a better English instruction does not work

This answers the underlying question better than the positive result does. Our
ER instruction — *"Find records that describe the same real-world entity as: "* —
is a genuinely better *description* of blocking than "retrieve relevant passages
that answer the query". It describes the task correctly. It is also, on the
models where it matters most, **actively harmful**:

| model | benchmark | `er_symmetric` Δ recall | 95% CI |
|---|---|---:|---|
| `intfloat/e5-base-v2` | wdc_computers | **−0.1038** | [−0.1238, −0.0815] |
| `BAAI/bge-base-en-v1.5` | wdc_computers | **−0.0775** | [−0.0990, −0.0567] |
| `all-MiniLM-L6-v2` (control) | abt_buy | **−0.0667** | [−0.0839, −0.0505] |
| `all-MiniLM-L6-v2` (control) | wdc_computers | **−0.0632** | [−0.0830, −0.0440] |
| `intfloat/e5-base-v2` | abt_buy | **−0.0436** | [−0.0564, −0.0314] |
| `google/embeddinggemma-300m` | wdc_computers | +0.0004 | [−0.0131, +0.0149] |

**The lever is the checkpoint's trained prefix, not better English.** That is the
measured claim, and it is all the data supports.

A tempting *mechanism* — that an untrained prefix displaces every vector by a
large common offset which crowds out record-specific signal — is **not supported
by these rows, and an earlier draft of this doc asserted it with numbers that do
not appear in the data.** Recomputed from the committed rows: our ER sentence
shifts vectors by `0.0511`–`0.3825`, while the models' own documented prompted
sides shift by `0.0111`–`0.3625`. Those ranges **overlap almost entirely**, so
displacement magnitude does not separate the helpful prompts from the harmful
ones. Gemma's `clustering` template has the largest shift of any *documented*
prompt (`0.3625`) *and* is documented *and* is the most damaging arm — three
properties the mechanism cannot jointly explain. (The largest shift anywhere in
the sweep is `0.3825`, from our own ER sentence on the control model — which is
neither the best nor the worst arm, so the ordering by displacement carries no
information about the ordering by recall.)

Whatever makes a prefix helpful, it is not how far it moves the vectors. Treat
the mechanism as **an open question**, not a finding. (Caught by automated review
on PR #252.)

**But the honest version of this has three tiers, not two.** Qwen3 forced the
distinction, and it is the more useful result:

| what you do | outcome | evidence |
|---|---|---|
| Use the checkpoint's **documented retrieval prompt** | **Reliable win.** Significantly positive on ≥1 benchmark for all four instruction-trained models; never significantly negative. | On `wdc_computers`: Gemma +0.0394, bge +0.1224, Qwen3 +0.0646 (all exclude 0). e5 is the weakest case — its `wdc_computers` +0.0152 closes on zero, so its evidence is `abt_buy` +0.0069 [+0.0010, +0.0128]. |
| Substitute **your own task text into its template shape** | **Coin flip, and model-specific.** Never beat the model's own default on candidate recall. | Qwen3 +0.0388 (helped, and the only non-documented arm clear of zero on 3/3); Gemma **−0.1589** (hurt badly) |
| Use a **raw English sentence outside any template** | **Reliably harmful.** | −0.1038 e5, −0.0907 Qwen3, −0.0775 bge, −0.0632 MiniLM (all `wdc_computers`) |

So the answer to *"can we pick good ones rather than guessing?"* is: **yes, by
taking the checkpoint's own retrieval prompt — not by writing a better one.**
Authoring your own task description is at best a wash against the model's
default and at worst catastrophic, and *which* it is cannot be predicted from
the text: our ER description is equally accurate prose in both the Qwen3 case
where it helped and the Gemma case where it cost 16 recall points.

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

### 4.5 Does #242's warning false-fire on the documented query-only recipes?

This matters because §4.4 says a one-sided recipe is *correct* for bge and Qwen3,
while PR #242 added a `VectorBlocker` warning about half-driven recipes. A
warning that fires on the correct configuration trains users to ignore warnings.

**Settled by construction, not by reading the changelog.** Each configuration was
built through the real public API (`SentenceTransformerEmbedder` → `FAISSIndex` →
`VectorBlocker`) with a log handler attached to
`langres.core.blockers.vector`:

| configuration | result |
|---|---|
| bge documented — `query_prompt` set, no `prompt_name` | **silent** |
| Qwen3 documented — `query_prompt` set, no `prompt_name` | **silent** |
| Gemma documented — `prompt_name="document"` + `query_prompt` | **silent** |
| the actual trap — `prompt_name="document"`, `query_prompt=None` | **WARNED** |

**No false positive, and the warning still fires on the case it exists for.** The
condition (`vector.py:557`) requires a bound `prompt_name` *and* `query_prompt is
None`; the documented query-only shape is the mirror image, so it cannot trigger.

This invariant is already regression-tested — `tests/core/blockers/
test_asymmetric_prompt_recipe.py::TestCoherenceWarning::
test_silent_when_only_the_query_side_is_driven` pins exactly this shape — so no
new test was added. The probe above confirms the existing coverage holds with the
real documented prompt strings rather than synthetic ones.
