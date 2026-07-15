"""Behaviour + smoke tests for the Peeters LLM-EM replication seam.

Harness/data code, so this exercises behaviour and the key edges (per the tiered
coverage policy) rather than every line. The one *against-their-published-data*
check (exact pair-set equality) needs the network and is marked ``integration``
(skipped by the fast ``not slow and not integration`` subset); the full F1 +
prompt round-trip replay lives in
``examples/research/peeters_llm_em_replication.py`` (186 MB archive download).
"""

import csv
from dataclasses import replace
from importlib import resources

import pytest

from langres.core.metrics import classify_pairs
from langres.data import peeters as P
from langres.data.peeters import (
    DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX,
    PeetersReplicationSpec,
    get_peeters_replication,
    gold_match_pairs,
    judgements_from_answers,
    list_peeters_replications,
    load_peeters_records,
    load_peeters_sample,
    parse_binary_answer,
    regenerate_sample_rows,
    render_prompt,
    render_sample_prompts,
    serialize_record,
)

#: Expected per-dataset sample shape (positives + 1000 sampled negatives).
_EXPECTED = {
    "abt-buy": {"positives": 206, "total": 1206, "left_prefix": "a", "right_prefix": "b"},
    "amazon-google": {"positives": 234, "total": 1234, "left_prefix": "a", "right_prefix": "g"},
}


# --------------------------------------------------------------------------- #
# Manifest / loader-factory
# --------------------------------------------------------------------------- #


def test_manifest_registers_the_two_verified_product_slices() -> None:
    assert list_peeters_replications() == ["abt-buy", "amazon-google"]


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_get_replication_returns_spec(name: str) -> None:
    spec = get_peeters_replication(name)
    assert spec.name == name
    assert spec.entity_noun == "Product"
    assert spec.task_prefix == DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX
    assert spec.left_prefix == _EXPECTED[name]["left_prefix"]
    assert spec.right_prefix == _EXPECTED[name]["right_prefix"]


def test_get_replication_unknown_raises_with_suggestion() -> None:
    with pytest.raises(KeyError) as exc:
        get_peeters_replication("amazon-googl")
    assert "amazon-google" in str(exc.value)


def test_register_duplicate_name_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        P.register(get_peeters_replication("abt-buy"))


# --------------------------------------------------------------------------- #
# Answer parser (exact replica of check_for_prediction)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Yes", 1),
        ("yes", 1),
        ("Yes.", 1),
        (" YES ", 1),
        ("Yes, they refer to the same product.", 1),
        ("No", 0),
        ("no", 0),
        ("No.", 0),
        ("They are not the same.", 0),
        ("", 0),
        ("I cannot determine.", 0),
        ("Y-E-S", 1),  # punctuation removed -> "yes"
    ],
)
def test_parse_binary_answer(answer: str, expected: int) -> None:
    assert parse_binary_answer(answer) == expected


@pytest.mark.parametrize("answer", ["ye-s", "ye.s", "y-e-s", "Ye's", "ye_s", "Y.E.S."])
def test_parse_binary_answer_delegates_to_canonical_parser(answer: str) -> None:
    """``parse_binary_answer`` is a thin int adapter over ``parse_binary_yes_no``.

    These six intra-word-punctuation cases are exactly where the two historical
    parsers disagreed; after unification both must agree and return MATCH (the
    paper's ``check_for_prediction`` deletes punctuation -> "yes").
    """
    from langres.core.matchers.llm_judge import parse_binary_yes_no

    assert parse_binary_answer(answer) == 1
    assert parse_binary_answer(answer) == int(parse_binary_yes_no(answer).decision)


# --------------------------------------------------------------------------- #
# Live LLM-judge seam: template + record_serializer + candidates (paid path)
# --------------------------------------------------------------------------- #


def test_build_llm_prompt_template_carries_placeholders() -> None:
    spec = get_peeters_replication("abt-buy")
    template = P.build_llm_prompt_template(spec)
    assert "{left}" in template
    assert "{right}" in template
    assert template.startswith(spec.task_prefix)


def test_make_record_serializer_reproduces_serialize_record() -> None:
    spec = get_peeters_replication("abt-buy")
    left_records, _right = load_peeters_records(spec)
    serializer = P.make_record_serializer(spec)
    an_id = next(iter(left_records))
    record = P.PeetersRecord(id=an_id, fields=left_records[an_id])
    assert serializer(record) == serialize_record(left_records[an_id], spec.serialization_fields)


def test_build_candidates_aligns_with_sample_order() -> None:
    spec = get_peeters_replication("abt-buy")
    candidates = P.build_candidates(spec)
    sample = load_peeters_sample(spec)
    assert len(candidates) == len(sample) == _EXPECTED["abt-buy"]["total"]
    for candidate, (left_id, right_id, _label) in zip(candidates, sample):
        assert candidate.left.id == left_id
        assert candidate.right.id == right_id
        assert candidate.blocker_name == "peeters_sample"


def test_llm_rendering_reproduces_archived_prompts_byte_exact() -> None:
    """The template+serializer an LLMMatcher uses reproduces the $0-validated prompt.

    LLMMatcher renders each pair as ``template.replace("{left}", serializer(left))``
    ``.replace("{right}", serializer(right))`` (literal substitution). This must
    equal ``render_sample_prompts``' prompt byte-for-byte, so the PAID run pays
    for the exact prompt the offline replay already validated at F1 95.15.
    """
    spec = get_peeters_replication("abt-buy")
    template = P.build_llm_prompt_template(spec)
    serializer = P.make_record_serializer(spec)
    candidates = P.build_candidates(spec)
    prompts = render_sample_prompts(spec)
    assert len(candidates) == len(prompts)
    for candidate, pair in zip(candidates, prompts):
        rendered = template.replace("{left}", serializer(candidate.left)).replace(
            "{right}", serializer(candidate.right)
        )
        assert rendered == pair.prompt


# --------------------------------------------------------------------------- #
# Serializer (per-field whitespace-token truncation, space-joined)
# --------------------------------------------------------------------------- #


def test_serialize_truncates_each_field_by_whitespace_tokens() -> None:
    record = {"name": "one two three four five", "price": "9.99"}
    assert serialize_record(record, (("name", 3), ("price", 5))) == "one two three 9.99"


def test_serialize_empty_field_leaves_a_trailing_space() -> None:
    # Load-bearing: their abt-buy template renders "'{name} {price}'"; an empty
    # price yields a trailing space before the closing quote in the archive.
    record = {"name": "sony turntable", "price": ""}
    assert serialize_record(record, (("name", 50), ("price", 5))) == "sony turntable "


def test_serialize_respects_field_order_amazon_google() -> None:
    record = {"manufacturer": "intuit", "title": "quickbooks 2007", "price": "38.99"}
    fields = (("manufacturer", 5), ("title", 50), ("price", 5))
    assert serialize_record(record, fields) == "intuit quickbooks 2007 38.99"


def test_serialize_missing_column_treated_as_empty() -> None:
    assert serialize_record({"name": "acme"}, (("name", 50), ("price", 5))) == "acme "


@pytest.mark.parametrize(
    ("value", "cap", "expected"),
    [
        # Internal double space: split(" ") keeps the "" token, so cap=2 stops
        # after "a" (split() would keep "a b" -> the whole point of byte-exactness).
        ("a  b c", 2, "a"),
        # Leading space is a token under split(" ") but dropped by split().
        (" x y", 2, "x"),
    ],
)
def test_serialize_uses_single_space_split_not_whitespace_split(
    value: str, cap: int, expected: str
) -> None:
    # Locks the load-bearing invariant that serialize_record splits on a single
    # space (str.split(" ")), matching MatchGPT — a regression to str.split()
    # would silently break the byte-exact prompt round-trip.
    assert serialize_record({"name": value}, (("name", cap),)) == expected


# --------------------------------------------------------------------------- #
# Prompt renderer
# --------------------------------------------------------------------------- #


def test_render_prompt_matches_archive_shape() -> None:
    prompt = render_prompt(
        "left text ",
        "right text 9.99",
        task_prefix=DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX,
        entity_noun="Product",
    )
    assert prompt == (
        "Do the two product descriptions refer to the same real-world product? "
        "Answer with 'Yes' if they do and 'No' if they do not.\n"
        "Product 1: 'left text '\n"
        "Product 2: 'right text 9.99'"
    )


def test_domain_complex_force_prefix_ends_with_newline() -> None:
    assert DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX.endswith("\n")
    assert "same real-world product" in DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX


# --------------------------------------------------------------------------- #
# Pair-set slice: regeneration, committed artifact, subset-of-test guarantee
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_regenerate_sample_rows_counts(name: str) -> None:
    rows = regenerate_sample_rows(get_peeters_replication(name))
    positives = sum(1 for *_ids, label in rows if label == 1)
    negatives = sum(1 for *_ids, label in rows if label == 0)
    assert len(rows) == _EXPECTED[name]["total"]
    assert positives == _EXPECTED[name]["positives"]
    assert negatives == 1000


def test_regenerate_downsamples_positives_when_above_cap() -> None:
    # abt-buy has 206 positives; capping at 100 exercises the positive-sampling
    # branch (both shipped datasets sit under the default 250 cap).
    spec = replace(get_peeters_replication("abt-buy"), max_positives=100)
    rows = regenerate_sample_rows(spec)
    positives = sum(1 for *_ids, label in rows if label == 1)
    assert positives == 100
    assert len(rows) == 1100


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_regeneration_is_deterministic(name: str) -> None:
    spec = get_peeters_replication(name)
    assert regenerate_sample_rows(spec) == regenerate_sample_rows(spec)


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_committed_artifact_matches_regeneration(name: str) -> None:
    # The tracked pair-list is exactly what regeneration produces (set AND order),
    # so committed data can never silently drift from the reproducer.
    spec = get_peeters_replication(name)
    regenerated = [
        (f"{spec.left_prefix}{left}", f"{spec.right_prefix}{right}", label)
        for left, right, label in regenerate_sample_rows(spec)
    ]
    assert load_peeters_sample(spec) == regenerated


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_sample_is_a_subset_of_the_test_split(name: str) -> None:
    # Justifies "slice, not new dataset": every sampled pair is a test.csv row
    # with the same label.
    spec = get_peeters_replication(name)
    text = resources.files(spec.dataset_package).joinpath(spec.source_test_file).read_text()
    test_pairs = {
        (int(r["ltable_id"]), int(r["rtable_id"]), int(r["label"]))
        for r in csv.DictReader(text.splitlines())
    }
    sample = set(regenerate_sample_rows(spec))
    assert sample <= test_pairs
    assert len(sample) == _EXPECTED[name]["total"]


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_load_peeters_sample_prefixes_ids_and_keeps_order(name: str) -> None:
    spec = get_peeters_replication(name)
    sample = load_peeters_sample(spec)
    assert len(sample) == _EXPECTED[name]["total"]
    # Positives are emitted before the sampled negatives, so the first row is a
    # match (the archive's JSONL line order relies on this).
    assert sample[0][2] == 1
    assert sample[-1][2] == 0
    assert all(lid.startswith(spec.left_prefix) for lid, _r, _l in sample)
    assert all(rid.startswith(spec.right_prefix) for _l, rid, _lab in sample)


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_load_peeters_records_keyed_by_prefixed_id(name: str) -> None:
    spec = get_peeters_replication(name)
    left, right = load_peeters_records(spec)
    # Every sampled id must resolve to a record.
    for lid, rid, _label in load_peeters_sample(spec):
        assert lid in left
        assert rid in right
    assert all(k.startswith(spec.left_prefix) for k in left)
    assert all(k.startswith(spec.right_prefix) for k in right)


# --------------------------------------------------------------------------- #
# Bridges: rendered prompts + replayed answers -> metric inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_render_sample_prompts_smoke(name: str) -> None:
    spec = get_peeters_replication(name)
    prompts = render_sample_prompts(spec)
    assert len(prompts) == _EXPECTED[name]["total"]
    assert prompts[0].label == 1
    first = prompts[0]
    assert first.prompt.startswith(DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX)
    assert f"{spec.entity_noun} 1:" in first.prompt
    assert f"{spec.entity_noun} 2:" in first.prompt


def test_judgements_from_answers_length_mismatch_raises() -> None:
    spec = get_peeters_replication("abt-buy")
    prompts = render_sample_prompts(spec)[:3]
    with pytest.raises(ValueError, match="align 1:1"):
        judgements_from_answers(prompts, ["Yes", "No"])


def test_replay_metric_wiring_with_oracle_answers() -> None:
    # A perfect oracle must reproduce a perfect score through the real metric
    # path (validates the judgements -> classify_pairs bridge, not sklearn).
    spec = get_peeters_replication("abt-buy")
    prompts = render_sample_prompts(spec)
    oracle = ["Yes" if p.label == 1 else "No" for p in prompts]
    judgements = judgements_from_answers(prompts, oracle)
    metrics = classify_pairs(judgements, gold_match_pairs(prompts), 0.5)
    assert metrics.tp == _EXPECTED["abt-buy"]["positives"]
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.f1 == pytest.approx(1.0)


def test_replay_metric_wiring_counts_one_miss_and_one_false_alarm() -> None:
    # Flip exactly one positive to "No" and one negative to "Yes"; the pair-level
    # metric must show fn=1, fp=1.
    spec = get_peeters_replication("abt-buy")
    prompts = render_sample_prompts(spec)
    answers = ["Yes" if p.label == 1 else "No" for p in prompts]
    first_pos = next(i for i, p in enumerate(prompts) if p.label == 1)
    first_neg = next(i for i, p in enumerate(prompts) if p.label == 0)
    answers[first_pos] = "No"
    answers[first_neg] = "Yes"
    judgements = judgements_from_answers(prompts, answers)
    metrics = classify_pairs(judgements, gold_match_pairs(prompts), 0.5)
    assert metrics.tp == _EXPECTED["abt-buy"]["positives"] - 1
    assert metrics.fp == 1
    assert metrics.fn == 1


# --------------------------------------------------------------------------- #
# Integration: exact equality against their published sampled_gs (network)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize("name", ["abt-buy", "amazon-google"])
def test_regenerated_sample_equals_published_pickle(name: str) -> None:
    """Our numpy regeneration reproduces their pandas ``sampled_gs`` exactly."""
    import gzip
    import urllib.request

    pd = pytest.importorskip("pandas")
    spec = get_peeters_replication(name)
    url = (
        "https://raw.githubusercontent.com/wbsg-uni-mannheim/MatchGPT/main/"
        f"LLMForEM/data/{name}/{name}-sampled_gs.pkl.gz"
    )
    left_source, right_source = name.split("-")  # e.g. "abt"/"buy", "amazon"/"google"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted host)
            frame = pd.read_pickle(gzip.GzipFile(fileobj=resp))
    except Exception as exc:  # pragma: no cover - network flakiness
        pytest.skip(f"could not fetch published sample: {exc}")

    theirs = {
        (
            int(row["id_left"].replace(f"{left_source}_", "")),
            int(row["id_right"].replace(f"{right_source}_", "")),
            int(row["label"]),
        )
        for _idx, row in frame.iterrows()
    }
    assert set(regenerate_sample_rows(spec)) == theirs
