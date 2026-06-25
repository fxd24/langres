# Dependency Security Policy

This project separates **hard controls** (the gates that actually protect the
supply chain) from **advisory visibility** (scanners that surface findings for
review but never block a merge).

## Hard Controls (the gates)

These are what actually defend the dependency set:

1. **7-day `exclude-newer` quarantine** — `[tool.uv] exclude-newer` in
   `pyproject.toml` makes `uv` ignore any version published less than 7 days
   ago, so a malicious upload has time to be detected and yanked before it can
   enter a resolution.
2. **Dependabot `cooldown: 7 days`** — mirrors the quarantine on Dependabot's
   path: it waits 7 days before raising a PR for a new version.
3. **Dependabot security PRs** — Dependabot still opens PRs for disclosed
   vulnerabilities, so fixes are surfaced and applied through review.
4. **`constraint-dependencies`** — the mechanism to hard-block a specific
   flagged version (see "Blocking a Flagged Version" below).

## Advisory Visibility (not merge gates)

`pip-audit` and `guarddog` — via `make security` and the Security workflow
(`.github/workflows/security.yml`) — are **advisory signals only**. They run on
every PR and on the weekly schedule, but their steps use `continue-on-error`
and do **not** block merges, because:

- The quarantine **deliberately delays patches**, so a non-empty CVE list is
  *expected*: recently-disclosed fixes are simply newer than the 7-day window
  and land on the next `make upgrade-deps` (quarantine roll), or sooner via a
  Dependabot security PR.
- `guarddog` is heuristic and produces many false positives (CUDA packages'
  bundled `.so` binaries, the `dspy` ↔ `dnspython` typosquatting false
  positive, single-file pure-Python packages, etc.).

Findings are reviewed on the **weekly scheduled run**, not gated per-PR.

| Command | Tool | What it checks |
|---------|------|----------------|
| `make audit` | pip-audit | Known CVEs in the synced environment (via PyPI/OSV) |
| `make scan-malware` | guarddog | Malicious indicators in package source (typosquatting, suspicious code patterns) |
| `make security` | both | Runs `audit` then `scan-malware` |

## Upgrading Dependencies

```bash
make upgrade-deps
```

This does two things atomically:
1. Rolls `exclude-newer` forward to today − 7 days in `pyproject.toml`.
2. Runs `uv lock --upgrade` to resolve the latest versions within that window.

Recently-disclosed CVE fixes that were newer than the previous quarantine
window get pulled in here. Commit the updated `pyproject.toml`. (`uv.lock` is
gitignored — langres is a library and CI resolves fresh via `uv sync` — so it
is not committed.)

## Blocking a Flagged Version

If a specific version is reported as malicious or broken, hard-block it in
`pyproject.toml` before re-locking by uncommenting and editing the example:

```toml
[tool.uv]
constraint-dependencies = ["somepkg!=1.2.3"]
```

Document the reason in a comment on the same line.

## Dependabot Configuration

`.github/dependabot.yml` configures `cooldown: default-days: 7` for the pip
ecosystem (mirroring the quarantine) and a separate weekly entry for the
`github-actions` ecosystem so action versions stay current.
