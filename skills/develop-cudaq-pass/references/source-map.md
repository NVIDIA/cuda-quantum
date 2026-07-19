# CUDA-Q Compiler Pass Source Map

Resolve every path from the CUDA-Q repository root. Read only the sections
needed for the current task. Human documentation explains the workflow and
semantics; this file only routes to current sources.

## Contributor Documentation

| Need | Path |
|---|---|
| Contribution and DCO policy | `Contributing.md` |
| Build, style, and test workflow | `Developing.md` |
| C++ API boundaries | `CppAPICodingStyle.md` |
| Compiler overview | `docs/sphinx/using/extending/compiler/index.rst` |
| Compiler IR forms | `docs/sphinx/using/extending/compiler/cudaq_ir.rst` |
| Dialect reference index | `docs/sphinx/using/extending/compiler/dialect_reference.rst` |
| Built-in pass workflow | `docs/sphinx/using/extending/compiler/mlir_pass.rst` |
| Available-pass catalog | `docs/sphinx/using/extending/compiler/available_passes.md` |
| External plugin workflow | `docs/sphinx/using/extending/compiler/pass_plugins.rst` |
| Quake reference and value semantics | `docs/sphinx/specification/quake-dialect.md` |

Load the compiler overview and pass guide for every pass-development task.
Load the plugin guide only for external work. Load the compiler IR guide and a
dialect reference only when the task depends on that IR boundary or dialect.
Load the Quake specification for Quake transformations and pipeline tasks that
operate on Quake.

## Generated References

Generated Markdown is not committed. `scripts/build_docs.sh` stages it under
`docs/sphinx/_mdgen/` only while building the documentation.

| Reference | Build output | Staged path |
|---|---|---|
| Transform passes | `build/docs/Transforms.md` | `docs/sphinx/_mdgen/Transforms.md` |
| CodeGen passes | `build/docs/CodeGenPasses.md` | `docs/sphinx/_mdgen/CodeGenPasses.md` |
| Quake dialect | `build/docs/Dialects/Quake.md` | `docs/sphinx/_mdgen/Dialects/Quake.md` |
| CC dialect | `build/docs/Dialects/CC.md` | `docs/sphinx/_mdgen/Dialects/CC.md` |
| QEC dialect | `build/docs/Dialects/QEC.md` | `docs/sphinx/_mdgen/Dialects/QEC.md` |
| CodeGen dialect | `build/docs/Dialects/CodeGen.md` | `docs/sphinx/_mdgen/Dialects/CodeGen.md` |

When the generated pass pages are absent, inspect both
`cudaq/include/cudaq/Optimizer/Transforms/Passes.td` and
`cudaq/include/cudaq/Optimizer/CodeGen/Passes.td`. When a generated dialect
page is absent, use the TableGen and implementation paths below.

## Pass Declarations and Registration

| Need | Path |
|---|---|
| General transformation declarations | `cudaq/include/cudaq/Optimizer/Transforms/Passes.td` |
| Code generation pass declarations | `cudaq/include/cudaq/Optimizer/CodeGen/Passes.td` |
| Transform factories and pipelines | `cudaq/include/cudaq/Optimizer/Transforms/Passes.h` |
| CodeGen factories and pipelines | `cudaq/include/cudaq/Optimizer/CodeGen/Passes.h` |
| CUDA-Q pass and pipeline registration | `cudaq/include/cudaq/Optimizer/InitAllPasses.h` |
| Pass driver registration call | `cudaq/tools/cudaq-opt/cudaq-opt.cpp` |

TableGen generates normal pass factories and registration. Inspect handwritten
headers or registration only when the generated surface is insufficient.

## Implementations, Pipelines, and Builds

| Need | Path |
|---|---|
| Transform implementations | `cudaq/lib/Optimizer/Transforms/` |
| Transform source list | `cudaq/lib/Optimizer/Transforms/CMakeLists.txt` |
| Shared transform pipelines | `cudaq/lib/Optimizer/Transforms/Pipelines.cpp` |
| CodeGen implementations | `cudaq/lib/Optimizer/CodeGen/` |
| CodeGen source list | `cudaq/lib/Optimizer/CodeGen/CMakeLists.txt` |
| Shared CodeGen pipelines | `cudaq/lib/Optimizer/CodeGen/Pipelines.cpp` |
| Optimizer library composition | `cudaq/lib/Optimizer/CMakeLists.txt` |
| Target-selected pipelines and options | `runtime/cudaq/platform/` |

Trace the producer and consumer of the relevant IR. A nearby pass name does not
establish ownership, and an existing pipeline is not necessarily a standard
optimization pipeline.

## Dialect Definitions and Implementations

| Dialect | TableGen and headers | Implementations |
|---|---|---|
| Quake | `cudaq/include/cudaq/Optimizer/Dialect/Quake/` | `cudaq/lib/Optimizer/Dialect/Quake/` |
| CC | `cudaq/include/cudaq/Optimizer/Dialect/CC/` | `cudaq/lib/Optimizer/Dialect/CC/` |
| QEC | `cudaq/include/cudaq/Optimizer/Dialect/QEC/` | `cudaq/lib/Optimizer/Dialect/QEC/` |
| CodeGen | `cudaq/include/cudaq/Optimizer/CodeGen/CodeGenDialect.td`, `CodeGenOps.td`, and `CodeGenTypes.td` | `cudaq/lib/Optimizer/CodeGen/CodeGenDialect.cpp`, `CodeGenOps.cpp`, and `CodeGenTypes.cpp` |

Inspect operation, type, interface, trait, verifier, parser, printer, and
bytecode definitions only when the task depends on them. A change to their
semantics is a prerequisite review unit rather than pass-local work.

## Tests and Validation

| Need | Path |
|---|---|
| lit configuration | `cudaq/test/lit.cfg.py` |
| Transformation regression tests | `cudaq/test/Transforms/` |
| Frontend Quake tests | `cudaq/test/AST-Quake/` |
| Translation tests | `cudaq/test/Translate/` |
| External plugin source and build test | `cudaq/test/plugin/CustomPassPlugin.cpp` and `cudaq/test/plugin/CMakeLists.txt` |
| Optimizer unit tests | `cudaq/unittests/Optimizer/` |
| Small unitary equivalence tool | `utils/CircuitCheck/` |
| Python MLIR regressions | `python/tests/mlir/` |

Use the smallest existing suite that owns the behavior. `CircuitCheck` is a
dense-unitary tool for small supported inputs; it does not establish
non-unitary correctness, output legality, optimization quality, or scaling.
