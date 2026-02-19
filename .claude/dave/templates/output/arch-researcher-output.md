<!-- TEMPLATE: dave-arch-researcher output
     PURPOSE: Deep codebase EVIDENCE for architectural decisions.
     DIFFERS FROM architect: Arch-researcher reads actual source code, traces
     integration points, and provides file-level evidence. Architect uses this
     evidence to compare design options at a system level. Arch-researcher is
     more granular (specific methods, imports, data flow). -->

# Architecture Research: {Phase Name}

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
**Structure:** {Brief description}
**New files:** {list}
**Modified files:** {list}

**Data flow:**
1. {step}
2. {step}

**Strengths:**
- {strength with evidence}

**Weaknesses:**
- {weakness with evidence}

**Tier 1 compliance:**
- [{ID}]: {compliant/not} -- {why}

### Option 2: {Name}
{Same structure}

### Option 3: {Name}
{Same structure}

## Comparison

| Criterion | Option 1 | Option 2 | Option 3 |
|-----------|----------|----------|----------|
| {criterion} | {value} | {value} | {value} |

## Recommendation

**Recommended:** Option {N} - {Name}

**Rationale:** {Why this option, grounded in codebase evidence}

**Why not alternatives:**
- Option {X}: {reason}
- Option {Y}: {reason}

**Implementation sequence:**
1. {step}
2. {step}

**Confidence:** {HIGH/MEDIUM/LOW} -- {rationale}

## Concerns

### {Concern 1}
- **What:** {description}
- **Impact:** {effect on the plan}
- **Mitigation:** {suggested approach}

## Integration Analysis

### Entry Points
- {Where the new code is called from}

### Dependencies
- {What the new code depends on}

### Downstream Effects
- {What existing code is affected by the new code}
