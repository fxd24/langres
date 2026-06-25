# Dependency Security Policy

## 7-Day Supply-Chain Quarantine

`pyproject.toml` contains:

```toml
[tool.uv]
exclude-newer = "YYYY-MM-DD"
```

This tells `uv` to ignore any package version published **after** the date shown.
The date is set to **today minus 7 days**, giving the community time to detect and
yank malicious uploads before they land in our lock file.

## Upgrading Dependencies

```bash
make upgrade-deps
```

This does two things atomically:
1. Rolls `exclude-newer` forward to today − 7 days.
2. Runs `uv lock --upgrade` to resolve the latest versions within that window.

Commit both `pyproject.toml` and `uv.lock` together.

## Blocking a Flagged Version

If a specific version is reported as malicious or broken, add it to
`pyproject.toml` before re-locking:

```toml
[tool.uv]
constraint-dependencies = ["somepkg!=1.2.3"]
```

Uncomment the commented-out example in `pyproject.toml` and set the package and
version. Document the reason in a comment on the same line.

## Vulnerability and Malware Scanning

Two tools are available as Make targets and in CI (`.github/workflows/security.yml`):

| Command | Tool | What it checks |
|---------|------|----------------|
| `make audit` | pip-audit | Known CVEs in the synced environment (via PyPI/OSV) |
| `make scan-malware` | guarddog | Malicious indicators in package source (typosquatting, suspicious code patterns) |
| `make security` | both | Runs `audit` then `scan-malware` |

CI runs both on every PR, push to `main`, and weekly on a schedule.

## Dependabot

`.github/dependabot.yml` is configured with `cooldown: default-days: 7` for the
pip ecosystem — Dependabot waits 7 days before raising a PR for a new version,
mirroring the `exclude-newer` quarantine. GitHub Actions versions are also kept
current via a separate weekly Dependabot entry.
