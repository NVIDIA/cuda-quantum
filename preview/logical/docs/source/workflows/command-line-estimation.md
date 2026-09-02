# Command-line estimation

Everything the Python front door does is also available as an explicit MLIR
pass pipeline driven by `qlx-opt`, with Stim text emitted by `qlx-translate`.
This route is useful when another compiler already owns the pass manager,
when inspecting intermediate IR, and in build or CI systems that do not want
to embed Python. Every command on this page is exercised verbatim by
`test/QLX/cli-workflows.test` against the fixtures under `examples/cli/`.

## Getting the tools

Build `cudaq.logical` against the `cudaq-devel` SDK as described in the
package README, then add the build tree's `bin` directory to `PATH`:

```bash
export PATH="$PWD/preview/logical/build/bin:$PATH"
```

`qlx-opt --help` is the authoritative list of registered CUDA-Q Logical
passes, and `qlx-translate --help` lists the registered translations. The
estimation-relevant entries:

| Tool | Pass / translation | What it produces |
|---|---|---|
| `qlx-opt` | `qlx-estimate-logical` | tier `LOGICAL`: a `qlx.estimate_result` over one verified P0 `qlx.program` |
| `qlx-opt` | `qlx-to-lvm` | a placed P1 `lvm.kernel` refined against a linked `lvm.domain` |
| `qlx-opt` | `fabric-count` | tier `STATIC`: a `qlx.estimate_result` over one selected P2 root and its `qlx.device` |
| `qlx-translate` | `fabric-to-stim` | Stim circuit text from a legalized P2 entry gadget |

Each estimation pass emits a symbol-addressable `qlx.estimate_result`
operation carrying a versioned `schema`, its `tier`, the `root` it counted,
and the typed payload. The result is an analysis artifact in the module — it
does not change any program's return type.

## Logical counts and native placement

`examples/cli/logical_and_placement.mlir` holds a P0 Bell program plus a
two-slot logical machine (`lvm.domain @machine`). One pipeline emits the
tier-`LOGICAL` estimate and refines the program to P1:

```bash
qlx-opt preview/logical/examples/cli/logical_and_placement.mlir \
  --pass-pipeline='builtin.module(qlx-estimate-logical{root=bell result=logical},qlx-to-lvm{root=bell domain=machine result=bell_placed})'
```

The output module retains the original P0 program and adds both products:

```mlir
qlx.estimate_result @logical {
  assumptions = ["dynamic branches are counted as a static upper bound"],
  data = {actions = {qlx_standard_cx = 1 : i64, qlx_standard_h = 1 : i64},
          instruments = {qlx_standard_measure_z = 2 : i64,
                         qlx_standard_prepare_zero = 2 : i64},
          logical_qubits_peak = 2 : i64, ...},
  evidence = [@bell],
  root = @bell,
  schema = "qlx.logical-profile/v1",
  tier = "logical"
}

lvm.kernel @bell_placed on @machine : () -> (i1, i1) attributes {
  input_p0 = @bell, qlx.placement_policy = "native-first-fit/v3",
  placement_witness_sha256 = "sha256:5986d81a...", qlx.stage = "p1"} { ... }
```

Every `qlx.estimate_result` carries an explicit `assumptions` list and an
`evidence` citation — the estimate states what it assumed and what it
counted, in the module, next to the numbers. The placed kernel records its
placement policy and a SHA-256 commitment to the detached placement witness.

The native `qlx-to-lvm` pass is a deterministic first-fit subset of the
richer Python placement solver (`qlx.compiler.place`): it requires an
explicit `lvm.domain`, rejects calls that have not been inlined, and fails if
control-flow joins do not agree on slots. Stay inside that shared subset on
the CLI, and use the Python route for research placement policies — both
produce the same verified P1 contract, and the CLI never silently falls back
to a Python policy.

## Static P2 counts

`examples/cli/static.mlir` holds a selected P2 memory gadget over the
Steane code — allocate one patch, apply H, idle three syndrome rounds,
deallocate — bound through a `qlx.device` to its logical machine. The code
carries a `fabric.code_profile` whose distance is `exact` with explicit
evidence. `fabric-count` produces the tier-`STATIC` estimate:

```bash
qlx-opt preview/logical/examples/cli/static.mlir \
  --pass-pipeline='builtin.module(fabric-count{root=memory device=device result=static})'
```

```mlir
qlx.estimate_result @static {
  assumptions = ["dynamic branches are counted as a static upper bound"],
  data = {operation_counts = {alloc = 1 : i64, dealloc = 1 : i64,
                              h = 1 : i64, idle = 1 : i64},
          patches_peak = 1 : i64,
          per_region = {compute = {code = "steane",
                                   rounds_by_kind = {idle = 3 : i64}, ...}},
          source_stage = "p2",
          source_facets = ["qec_spec", "qec_realization"], ...},
  device = @device,
  evidence = [@memory],
  root = @memory,
  schema = "qlx.fabric-counts/v1",
  tier = "static"
}
```

The folded `fabric.idle {rounds = 3}` is counted once with its multiplicity
visible in `rounds_by_kind` — the IR is never unrolled for counting.

The pass requires the selected P2 root and the device it was realized
against; both must be present in the module, and the module must pass
native verification. There is no tier inference from filenames or partial
annotations — a missing root or device is a hard error.

## Fail-closed by construction

The CLI does not guess missing facts, reinterpret earlier stages, or emit a
partial result after a failed check. The two estimators above are exactly
the tiers the Python `qlx.estimate` front door computes; the pass options
(`root=`, `result=`, `device=`, `domain=`) only make explicit what Python
infers from the build.

For turning a selected P2 program into Stim circuit text, continue to
[Stim emission](stim-emission).
