---
name: develop-cudaq-pass
description: "Use when planning, adding, modifying, debugging, hardening, or integrating a CUDA-Q compiler pass, whether built into CUDA-Q or supplied as an external plugin. Covers transformations, analyses, conversions, code generation, and pipeline placement. Do not use for merely invoking existing passes, runtime optimization, or general MLIR guidance unrelated to CUDA-Q."
---

# Develop a CUDA-Q Compiler Pass

Use the current CUDA-Q source and compiler contributor documentation to plan or
implement one pass-sized change. Keep human documentation authoritative and use
the [source map](references/source-map.md) to find it.

## Load Sources Progressively

1. Classify the work as a built-in pass or an external plugin. Read the
   compiler overview and pass guide. Read the plugin guide only for an external
   pass.
2. Review the generated available-pass catalog before deciding whether to
   extend or add a pass. If its generated includes are absent, inspect both the
   Transform and CodeGen `Passes.td` files.
3. Read the compiler IR guide when the task depends on an IR boundary. Read
   only the generated dialect references used by the task. When a generated
   dialect page is absent, inspect its TableGen definitions and implementations
   instead.
4. For a Quake transformation, or a pipeline task that operates on Quake, read
   the Quake semantic specification. Do not assume Quake has one global
   canonical form.
5. Trace the closest comparable declaration, implementation, registration,
   CMake entry, pipeline use, and tests. Load a named primary source when the
   requested algorithm depends on one.

## Choose the Change Shape

Before choosing the change shape, capture representative valid input IR at the
intended pipeline boundary and state the output required by the next consumer.

Select the smallest extension point that owns the behavior:

- Use folding or operation canonicalization for a cheap, local, monotonic form
  that is valid whenever the operation appears.
- Use an analysis for read-only facts with defined invalidation. Keep a one-use
  calculation private until it represents a stable reusable compiler concept.
- Reuse an MLIR interface first. Add an interface or trait only for intrinsic
  behavior needed by multiple operations or consumers.
- Add IR only when current operations, types, attributes, or interfaces cannot
  state or verify a required semantic distinction. Treat this as a prerequisite
  review unit.
- Extend an existing pass only when the work is part of the same named
  transformation and retains its purpose, anchor, input and output IR,
  placement, enablement, convergence model, and performance budget.
- Otherwise add a focused pass. Keep pass-dependent patterns and helpers local;
  share a helper only when it has more than one real caller.

`quake-simplify` contains existing transformations but is not the default home
for new independent ones.

## Prepare the Implementation Brief

After tracing current behavior, present a proportional brief containing:

- built-in or external ownership, selected change shape, and the rejected
  nearest alternative
- approach, pass anchor, accepted input IR, produced output IR, and behavior
  for valid but unsupported input
- computation or observable behavior preserved by the transformation
- ordered proposed review units and dependencies
- minimum sufficient test evidence
- non-obvious correctness, compile-time, memory, convergence, or production
  risks
- likely files and the repository that owns each change

Separate prerequisite shared APIs, analyses, utilities, traits, interfaces, or
IR semantics from pass-local implementation and pipeline activation. A local
implementation does not establish that required owner review occurred.

For an external plugin, keep its implementation, packaging, and tests in the
plugin repository and follow that repository's contribution process. Put a
required CUDA-Q shared API or IR change in a separate CUDA-Q review unit and
follow CUDA-Q's contribution process for that unit.

If the contributor requests a plan, design, proposal, or no edits, stop after
the brief. Otherwise continue without another confirmation gate unless a
stopping condition below applies.

## Communicate the Work

Apply these rules to briefs, progress reports, completion evidence,
diagnostics, comments, and documentation:

- Lead with the decision, result, action, or blocking question and provide the
  evidence and rationale needed to act.
- Use established CUDA-Q, MLIR, compiler, and quantum computing terms. Define
  necessary specialized terms and write for an entry-level quantum compiler
  developer without assuming local history or an unstated pipeline.
- Prefer exact names, paths, commands, results, and measurements. Distinguish
  verified facts, recommendations, assumptions, and unresolved decisions.
- Preserve relevant preconditions, correctness risks, and tradeoffs without
  inflated claims or unnecessary context.
- Do not repeat the prompt, tool logs, repository policy, or source material.

## Preserve Quantum Semantics

For every quantum transformation, state and justify one applicable relation:

- exact equivalence
- equivalence up to global phase
- equivalence under a known qubit mapping
- bounded approximation with a named error measure and tolerance
- preserved non-unitary observable behavior

Verifier-valid IR and an improved gate count, depth, or other optimization
metric are not evidence of semantics preservation. Use `CircuitCheck` only for
small unitary IR within its supported scope and select its global-phase or
mapping modes only when the pass claims that relation. For measurement, reset,
control flow, noise, or other non-unitary behavior, test the observable behavior
that the pass claims to preserve.

For Quake dependency-sensitive transformations, prefer value form when precise
wire or cable use-def chains are required. Preserve linear value threading
through operations and control flow. Treat calls and unsupported regions
conservatively unless a pass-specific analysis establishes otherwise.

## Implement and Verify

1. Add or reproduce the smallest failing test at the layer that owns the
   behavior.
2. Implement through the appropriate MLIR rewrite, conversion, analysis, or
   ordered traversal mechanism. Match before mutation, use `PatternRewriter`
   for pattern rewrites, avoid global mutable state, and keep output and
   diagnostics deterministic.
3. Build only affected targets first. Run focused lit, unit, plugin, pipeline,
   target, or frontend checks justified by the changed contract.
4. For a quantum rewrite, check both the intended IR change and the claimed
   semantic relation. Test pipeline composition separately from direct pass
   behavior.
5. Format changed code, inspect the complete diff, and run justified repository
   checks. Do not construct a Cartesian test matrix.

Use lit and FileCheck for pass IR, options, diagnostics, and unsupported input.
Use optimizer unit tests for reusable analyses, algorithms, and data
structures. Add pipeline or target tests only when placement or integration
changes observable behavior. Measure credible scaling risks with a
reproducible fixture; do not create a persistent benchmark without an agreed
home.

## Stop and Report

Pause when implementation needs a contributor decision about undefined
semantics, a shared or core IR change, pipeline placement, or the home for a
persistent performance test.

When finished, group changes and evidence by review unit. Report tests run and
omitted, remaining risks, and unresolved decisions. Follow the repository's
current contribution process.
