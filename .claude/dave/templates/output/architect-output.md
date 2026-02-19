<!-- TEMPLATE: dave-architect output
     PURPOSE: Multi-option architectural comparison focused on DESIGN DECISIONS.
     DIFFERS FROM arch-researcher: Architect evaluates options at a higher level,
     focusing on how features FIT into the system. Arch-researcher provides deep
     codebase EVIDENCE (specific files, methods, integration points) that grounds
     the architect's options. -->

# Architecture & Design Research: {Phase Name}

## Codebase Exploration Summary

### Relevant Existing Code
| Component | File | Role | Key Methods |
|-----------|------|------|-------------|
| {name} | {path} | {what it does} | {methods relevant to this feature} |

### Patterns Observed
- **{Pattern name}:** {How it is used, where it appears, why it matters}

### Existing Utilities Available
- {Utility}: {What it provides, why the new feature should use it}

## Tier 1 Constraints
| ID | Rule | Relevance to This Feature |
|----|------|--------------------------|
| {ID} | {rule text} | {how it constrains the design} |

## Architectural Options

### Option 1: {Name}
**Structure:** {description with file paths and class names}
**Data flow:** {step-by-step}
**Strengths:** {with evidence}
**Weaknesses:** {with evidence}
**Tier 1 compliance:** {per constraint}

### Option 2: {Name}
{Same structure}

### Option 3: {Name} (if applicable)
{Same structure}

## Comparison
| Criterion | Option 1 | Option 2 | Option 3 |
|-----------|----------|----------|----------|

## Recommendation
**Recommended:** Option {N} - {Name}
**Rationale:** {grounded in codebase evidence}
**Why not alternatives:** {per option}
**Implementation sequence:** {ordered steps}
**Confidence:** {HIGH/MEDIUM/LOW} — {rationale}

## Concerns
### {Concern}
- **What:** {description}
- **Impact:** {effect on the plan}
- **Mitigation:** {suggested approach}

## Integration Analysis
- **Entry points:** {where new code is called from}
- **Dependencies:** {what new code depends on}
- **Downstream effects:** {what existing code is affected}
