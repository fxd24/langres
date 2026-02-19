# Codebase Investigation Rules

Shared rules for agents that investigate the codebase (dave-architect, dave-arch-researcher).

## Critical Rules

**ALWAYS read actual source code.** Never guess about class names, method signatures, file locations, or import paths. If you state "class X has method Y," you must have read the file.

**ALWAYS trace integration points.** Do not assume how existing code works. Read the interface, read the implementation, check consumers.

**ALWAYS check each option against every Tier 1 constraint.** This is not optional. List each constraint and its compliance status explicitly.

**ALWAYS propose concrete structures.** "Create a service" is useless. "Create `[Name]Service` in `src/services/[domain]/[feature]_service.py` with method `async def [action](self, [params]) -> [ReturnType]`" is useful — include actual file paths, class names, and method signatures.

**ALWAYS include weaknesses for every option.** If you cannot find weaknesses, you have not thought critically enough.

**ALWAYS ground recommendations in codebase evidence.** "Option 1 is better because it follows the pattern established in [existing service] (src/services/[path])" is grounded. "Option 1 is cleaner" is not.

**NEVER propose patterns that contradict existing conventions** without explicitly flagging the deviation and justifying why.

**NEVER invent code that does not exist.** If you reference "the existing BaseGateway class," you must have found it.

**NEVER present only one option.** Minimum 2 options, ideally 3.

**NEVER ignore the existing codebase in favor of textbook patterns.** Consistency with the codebase is more important than theoretical perfection.

**Handle the "not found" case honestly.** "I could not find an existing pattern for X" is valuable — it tells downstream agents this is new territory.
