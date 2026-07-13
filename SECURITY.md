# Security Policy

## Supported versions

langres is pre-1.0; only the **latest release** on PyPI receives fixes.

## Reporting a vulnerability

Please report vulnerabilities **privately** — do not open a public issue:

- Preferred: [GitHub private vulnerability reporting](https://github.com/fxd24/langres/security/advisories/new)
  (Security tab → "Report a vulnerability").
- Or email **hire@grafdavid.com** with a description, reproduction steps,
  and impact.

You should receive an acknowledgement within a few days. Please allow a
reasonable window for a fix before public disclosure.

## Known, documented risk: prompt injection via record content

When an LLM-based judge is used (the default `"auto"` / `"zero_shot_llm"`, or
`LLMJudge` / `DSPyJudge` directly), the **content of the records being
compared is sent to the model**, so crafted field values can influence
verdicts. This is a documented design property, not a reportable
vulnerability — see
[README → Known limitations & security notes](README.md#known-limitations--security-notes)
for the details and mitigations (structured-output parsing limits but does not
eliminate it; do not feed untrusted third-party record content to an LLM judge
without review; the `"string"` / `"embedding"` judges are unaffected).

## Scope notes

- API keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`) are read from the
  environment and used for paid LLM calls; treat them like any credential.
  Verdict logs (`JudgementLog`) default to a privacy-safe row shape
  (ids/scores, `features=False`), but review what you persist.
- Resolver artifacts are **config-registry JSON, not pickle** — loading an
  artifact does not execute arbitrary code by design. If you find a way to
  make it do so, that *is* a reportable vulnerability.
