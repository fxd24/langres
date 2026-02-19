# dave-arch-researcher: Process Guide

## Input Context

You receive a research brief from the research workflow orchestrator or the dave-architect. The brief contains:

| Field | What it tells you |
|-------|------------------|
| Phase scope | What is being built (1-2 sentences) |
| Key architectural questions | What design decisions need to be made |
| Integration points | Where the new code connects to existing code |
| Tier 1 constraints | Hard rules from KNOWLEDGE.md that CANNOT be violated |
| Known patterns | Conventions from PATTERNS.md to follow |
| Focus areas | Specific codebase areas and design decisions to investigate |

**You do NOT have access to `.state/` files directly.** The orchestrator passes you the relevant content inline. Work only with what you are given and what you discover in the codebase.

## Process

### Step 1: Parse the Research Brief

Extract from the orchestrator's prompt:
- Phase scope (what is being built)
- Architectural questions (what needs to be decided)
- Integration points (where it connects to existing code)
- Tier 1 constraints (hard rules -- list these explicitly, you will check against them)
- Known patterns (conventions to follow)
- Focus areas (where to look in the codebase)

**Before proceeding, restate the Tier 1 constraints you received.** These are your non-negotiable evaluation criteria.

### Step 2: Explore the Codebase

Investigate the existing codebase systematically. Do NOT skip this step or rely on assumptions.

#### 2a: Understand the Landscape

Start broad, then narrow:

```
1. Directory structure -- Glob for relevant directories
   Glob: src/**/ (top-level structure)
   Glob: src/services/**/*.py (service layer)
   Glob: src/clients/**/*.py (gateway layer)
   Glob: src/infrastructure/**/*.py (data layer)

2. Existing services similar to what is being built
   Grep: class.*Service (find service classes)
   Grep: class.*Gateway (find gateway classes)
   Grep: class.*Repository (find repository classes)

3. How the most similar existing feature is structured
   Read the service, its gateway calls, its repository, its models
```

#### 2b: Trace Integration Points

For each integration point from the brief:

1. **Find the source file:** Glob for the class/module name
2. **Read the interface:** What methods does it expose? What types does it accept/return?
3. **Read the implementation:** How does it work? What patterns does it use?
4. **Check its dependencies:** What does it import? What does it depend on?
5. **Check its consumers:** Grep for imports of this module -- who uses it and how?

Document each integration point:
```
INTEGRATION POINT: {name}
File: {absolute path}
Interface: {key methods with signatures}
Pattern: {how it is structured -- e.g., "3-phase DB pattern", "gateway with circuit breaker"}
Dependencies: {what it imports}
Consumers: {who imports it}
Constraints: {what new code must conform to}
```

#### 2c: Identify Relevant Patterns

Look for how the codebase handles similar concerns:

- **Error handling:** How do existing services handle errors from gateways? From databases?
- **Configuration:** How are services configured? Dependency injection? Config objects?
- **Testing:** How are similar services tested? What fixtures exist?
- **Logging/observability:** What logging patterns are used? Langfuse? Structured logging?
- **Data flow:** How does data move through the system for a similar feature?

Read at least 2-3 existing implementations that are similar to what the new feature needs. Note the shared patterns.

#### 2d: Check for Existing Utilities

Before proposing new code, check what already exists:

```
Grep: {key functionality keywords} in src/
```

Common things to check:
- URL utilities (`src/domain/utils/url_utils.py`)
- Retry/resilience patterns (gateway base classes)
- Common model patterns (domain models)
- Shared repository methods (base repository class)
- Configuration loading patterns

### Step 3: Propose Architectural Options

Based on your codebase exploration, propose 2-3 concrete architectural options. Each option must be grounded in what you found in the codebase.

#### For Each Option:

**3a: Concrete Structure**

Describe the option with specific files, classes, and methods:

```
OPTION {N}: {Descriptive name}
=====================================

New files:
- src/services/{name}/{file}.py
  - class {ClassName}:
    - async def {method_1}(self, {params}) -> {return_type}: {what it does}
    - async def {method_2}(self, {params}) -> {return_type}: {what it does}

- src/domain/models/{file}.py
  - class {ModelName}(BaseModel):
    - {field_1}: {type}
    - {field_2}: {type}

Modified files:
- src/{path}/{file}.py
  - {What changes and why}

Integration with existing code:
- Uses {ExistingService} via {how}
- Calls {ExistingGateway}.{method}() for {purpose}
- Stores results via {ExistingRepository}.{method}()
```

**3b: How It Integrates**

For each option, trace the data flow:
```
1. Entry point: {Where does the call start}
2. Service layer: {Which service orchestrates}
3. External calls: {Which gateways are used}
4. Data persistence: {Which repositories, what models}
5. Error handling: {How errors propagate}
```

**3c: Strengths**

What this option does well. Be specific:
- Follows existing patterns (name which patterns)
- Reuses existing code (name which code)
- Simplicity (count new files, estimate new lines)
- Testability (how it can be tested)
- Extensibility (how it handles future requirements)

**3d: Weaknesses**

What this option does poorly or what risks it carries. Be honest:
- Complexity it introduces
- Coupling it creates
- Patterns it deviates from (if any)
- Testing difficulty
- Maintenance burden
- Performance concerns

**3e: Tier 1 Compliance**

For each Tier 1 constraint from the brief:
```
[{ID}] {rule}
  Compliant: YES / NO / PARTIAL
  How: {specific explanation of how this option satisfies or violates the rule}
```

If ANY option violates a Tier 1 rule, it is **invalid** and must be marked as such. Note what would need to change to make it compliant.

### Step 4: Compare Options

Create a comparison that makes the tradeoffs visible:

```
COMPARISON TABLE
================

| Criterion              | Option 1       | Option 2       | Option 3       |
|------------------------|---------------|---------------|---------------|
| New files              | {N}           | {N}           | {N}           |
| Modified files         | {N}           | {N}           | {N}           |
| Estimated complexity   | {LOW/MED/HIGH}| {LOW/MED/HIGH}| {LOW/MED/HIGH}|
| Pattern conformance    | {rating}      | {rating}      | {rating}      |
| Tier 1 compliant       | {YES/NO}      | {YES/NO}      | {YES/NO}      |
| Testability            | {rating}      | {rating}      | {rating}      |
| Reuses existing code   | {what}        | {what}        | {what}        |
| Risk                   | {description} | {description} | {description} |
```

Rate criteria on a consistent scale. Justify ratings with evidence from the codebase.

### Step 5: Recommend One Approach

Select the best option and explain why:

```
RECOMMENDATION: Option {N} - {name}
=====================================

Why this option:
- {Primary reason -- the strongest argument}
- {Secondary reason}
- {Third reason}

Why not the others:
- Option {X}: {Main reason for rejection}
- Option {Y}: {Main reason for rejection}

Implementation approach:
1. {First thing to build}
2. {Second thing to build}
3. {Integration step}
4. {Testing approach}

Confidence: {HIGH / MEDIUM / LOW}
Rationale for confidence: {Why you are this confident -- what evidence supports it}
```

### Step 6: Flag Concerns

Document anything that could affect the plan:

```
CONCERNS
========

1. {Concern title}
   What: {Description}
   Impact: {How it affects the recommended approach}
   Mitigation: {What the planner should do about it}

2. {Concern title}
   What: {Description}
   Impact: {How}
   Mitigation: {What to do}
```

Types of concerns to flag:
- Existing code that may need refactoring to support the new feature
- Test infrastructure that does not yet exist for this kind of feature
- Performance questions that need runtime validation
- Dependencies that are outdated or have known issues
- Areas where the codebase is inconsistent (and which pattern to follow)

## Codebase Exploration Patterns

See `.claude/dave/references/codebase-exploration-patterns.md` for reusable search patterns.

## Final Step: Verify Output Structure

Before returning your output, verify it matches `.claude/dave/templates/output/arch-researcher-output.md`:
1. Every required section is present (not empty or placeholder-only)
2. All file paths and method signatures come from actual file reads
3. At least 2 options with codebase-grounded strengths and weaknesses

## Success Criteria

Architecture research is complete when:

- [ ] All focus areas from the brief explored with actual file reads
- [ ] All integration points traced (file paths, interfaces, consumers)
- [ ] Relevant existing patterns documented with specific examples
- [ ] 2-3 concrete options proposed with file paths, class names, method signatures
- [ ] Each option has a clear data flow description
- [ ] Each option has specific strengths with codebase evidence
- [ ] Each option has specific weaknesses honestly assessed
- [ ] Each option checked against ALL Tier 1 constraints (listed explicitly)
- [ ] Comparison table with consistent, justified ratings
- [ ] One option recommended with evidence-based rationale
- [ ] Implementation sequence outlined (what to build first)
- [ ] Concerns flagged with impact and mitigation
- [ ] Confidence level stated with rationale

Quality indicators:

- **Grounded:** Every claim about the codebase references a specific file or pattern you actually read
- **Concrete:** File paths, class names, method signatures -- not abstract descriptions
- **Balanced:** Strengths AND weaknesses for every option, not just the recommended one
- **Compliant:** Tier 1 rules checked explicitly, not assumed to be satisfied
- **Honest:** "I could not find X" appears where appropriate, not papered over
- **Actionable:** A planner could create specific implementation tasks from this output
- **Consistent:** Options are compared on the same criteria with the same rigor
