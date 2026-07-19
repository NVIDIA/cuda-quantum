---
name: develop-cudaq-pass
description: "Use when planning, adding, modifying, debugging, hardening, or integrating a CUDA-Q compiler pass, whether built into CUDA-Q or supplied as an external plugin. Covers transformations, analyses, conversions, code generation, and pipeline placement. Do not use for merely invoking existing passes, runtime optimization, or general MLIR guidance unrelated to CUDA-Q."
---

# Develop a CUDA-Q Compiler Pass

Use the current CUDA-Q source and compiler contributor documentation to plan or
implement one pass-sized change. Keep human documentation authoritative and use
the [source map](references/source-map.md) to find it. Do not copy repository
policy or dialect reference material into the result.

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

Do not use `quake-simplify` as the default home for an independent
transformation. CUDA-Q has concrete compilation pipelines but no single
standard Quake optimization pipeline. Do not invent one as part of a pass.

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
- likely files and applicable routing from the current `.github/CODEOWNERS`

Separate prerequisite shared APIs, analyses, utilities, traits, interfaces, or
IR semantics from pass-local implementation and pipeline activation. A local
implementation does not establish that required owner review occurred.

For an external plugin, keep its implementation, packaging, and tests in the
plugin repository and use that repository's review routing. Put a required
CUDA-Q shared API or IR change in a separate CUDA-Q review unit and use CUDA-Q's
current `.github/CODEOWNERS` for that unit.

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
mapping modes only when the pass claims that relation. Use contract-specific
tests for measurement, reset, control flow, noise, or other non-unitary
behavior.

For Quake dependency-sensitive transformations, prefer value form when precise
wire or cable use-def chains are required. Preserve linear value threading
through operations and control flow. Treat calls and unsupported regions
conservatively unless a pass-specific analysis establishes otherwise. Do not
use `repair-linear-type` as evidence that a rewrite was sound.

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

Stop for contributor or owner input when semantics are undefined, a shared or
core IR requirement is unresolved, the required pipeline boundary is unknown,
or a persistent performance test has no agreed home.

The completion report groups changes and evidence by proposed review unit. It
names tests run and omitted, production findings, remaining risk, and required
human review. Follow the current repository contribution policy for commits
and remote actions.
