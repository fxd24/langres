# Confidence Level Calibration

This document defines confidence levels used across Dave Framework agents for findings, recommendations, and knowledge entries. All agents that assign confidence levels should reference this document.

## Confidence Levels

### HIGH
- **Definition:** Verified from authoritative sources and confirmed through evidence
- **Evidence requirement:** Official documentation URL + at least one corroborating source (GitHub release, source code, verified benchmark)
- **Examples:**
  - Library version confirmed from PyPI/npm release page
  - API behavior verified from official docs + tested in codebase
  - Performance claim backed by published benchmark + reproduced locally
- **Use when:** You can point to a specific authoritative source AND have corroborating evidence

### MEDIUM
- **Definition:** Supported by reliable sources but not fully verified
- **Evidence requirement:** At least one credible source (official docs, reputable technical blog, GitHub issue with maintainer response)
- **Examples:**
  - Best practice from official guide, not yet tested in this codebase
  - Pattern observed in 2+ similar projects but not verified here
  - Community consensus from GitHub discussions with maintainer acknowledgment
- **Use when:** You have a credible source but haven't independently verified the claim

### LOW
- **Definition:** Inferred, based on training data, or from unverified sources
- **Evidence requirement:** State the basis for the inference explicitly
- **Examples:**
  - Based on training data knowledge (may be stale)
  - Single blog post without official backing
  - Reasonable inference from related patterns but no direct evidence
  - Community forum answer without official confirmation
- **Use when:** You believe something is likely true but cannot point to authoritative evidence

## Review Aggregator Thresholds

For the review aggregator specifically:
- **Fix now:** >=80% confidence + critical/high severity
- **Defer:** >=60% confidence + medium severity
- **Dismiss:** <40% confidence OR contradicts Tier 1 knowledge
- **Open question:** 40-70% confidence with ambiguous evidence

## Multi-Reviewer Consensus

Consensus across reviewers adjusts confidence:
- **3+ reviewers flag same issue:** HIGH confidence regardless of individual evidence
- **2 reviewers flag same issue:** MEDIUM confidence minimum
- **1 reviewer only:** Use individual evidence quality

## Staleness

Training data knowledge is inherently LOW confidence because it may be 6-18 months stale. Always prefer current web sources over training data for version-sensitive claims.
