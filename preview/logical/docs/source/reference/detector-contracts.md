# Detector contracts and Clifford projection

A gadget has a logical action, a realization, encoded input and output
boundaries, and an affine outcome map. `implements=` names the action, while
the gadget body describes its realization. A detector contract describes how realization
measurements and virtual boundary syndromes determine detector parities and
output virtual syndromes.

## Gadget terminology

An **action** is the ideal logical circuit a gadget claims to implement. It can
prepare or discard qubits and return measurement outcomes; it need not be a
unitary operation. A **realization** is the circuit over the gadget's physical
code blocks. The affine **outcome map** converts realization outcomes into
action outcomes, which are the gadget's logical observables.

These concepts are separate from the **gadget detector contract**, which fixes
the virtual detectors, output virtual-syndrome relations, and input/output
**code presentations**. A code presentation selects stabilizer generators and
boundary-error coordinates. `GadgetContract` and `CodeProfile` are compiler
containers supplying those mathematical objects; their other fields do not
change the definitions.

Use `@cudaq.logical.action` to author a reusable ideal circuit. Its default
`kind="auto"` infers whether the circuit is unitary or an instrument.
`kind="unitary"` requires a body of unitary actions, and `kind="instrument"` permits
preparation, measurement, and changed quantum boundaries. `CliffordAction`
remains the narrower signed symplectic description of a Clifford unitary.
The verification receipt's `logical_action` reports the check of the declared
action. The verifier supports signed Clifford unitaries, single-logical
signed X/Z eigenstate preparation, and destructive or preserving X/Z
measurement.
Other objective forms are explicitly unsupported.

### Compatibility spellings

The former `objective` vocabulary remains accepted at compatibility boundaries:

| Canonical authoring name | Accepted compatibility name |
| --- | --- |
| `@cudaq.logical.action` | `@cudaq.logical.objective` |
| `kind="unitary"` | `kind="action"` |
| `ProgramDefinition.action_kind` | `ProgramDefinition.objective_kind` (the explicit unitary kind uses the legacy value `"action"`) |
| `ActionOperandRef`, `ActionOperands` | `ObjectiveOperandRef`, `ObjectiveOperands` |
| `ActionMismatch`, `MissingAction` | `ObjectiveMismatch`, `MissingObjective` |
| `logical.ActionFamily`, `logical.SyndromeExtractionAction` | `logical.ObjectiveFamily`, `logical.SyndromeExtractionObjective` |
| `CompilationContext.action`, `declare_action` | `objective`, `declare_objective` |
| QEC lowering `action=`, `action_family=` and corresponding read-only views | `objective=`, `objective_family=` and the stored fields |

These are alternate spellings of the same values, rather than separately
registered actions. Passing both keyword spellings is an error. The public
definition kind is `"action"`; constructing a definition with the legacy
`kind="objective"` normalizes it to that kind.

Saved MLIR and build bundles keep their existing `qlx.objective_body`,
`fabric.objective`, `objective_kind`, `objective_family`, and logical-link
`objective` spellings. Python and C++ expose `ActionBodyOp` and Fabric
`ActionOp` aliases for these operations. Provider manifest identities also
retain their existing wire names. These format choices preserve replay and
linkage without requiring a serialized-schema migration.

Placement and scheduling **optimization objectives** describe costs or solver
policies. Their `objective` names have that separate meaning and are unchanged.

## Virtual syndromes and boundary errors

These quantities describe different information at an encoded boundary:

| Quantity | Meaning |
| --- | --- |
| Measurement outcomes `M` | Physical realization measurement bits, in execution order. |
| Virtual syndrome `V` | Stabilizer signs expected from the available classical information. |
| Boundary error `E` | Residual phase-free Pauli error relative to the ideal output on the same outcome branch. |
| Syndrome coordinate `S` | Commutation of the boundary error with the code presentation's listed stabilizer generators. |
| Logical-effect coordinate `L` | Logical Pauli effect of that boundary error in the chosen coordinates. |

`V` is a classical expectation; `S` is a coordinate of a residual Pauli error.
A change in an inferred stabilizer sign is written `ΔV`, not `S`. The
code-presentation map sends `E` to `(S, L)`. A code presentation may list
redundant generators, so its attainable coordinates need not fill the whole
declared bit space.

The contract spellings `endpoint.syndrome`, `InputSyndromeRef`, and
`OutputSyndromeAssignment` describe **virtual syndromes**. They do not declare
residual-error coordinates. `CodeProfile` remains the compiler's larger code
algebra container; its selected generators and boundary coordinates supply a
code presentation. Contract syndrome bundles currently use the endpoint's
effective stabilizer basis.

## Detector map and outcome map

The affine detector map has the form

```text
DM: (M, V_in) -> (D, V_out)
```

`Detector` declares one virtual-detector parity. Output syndrome bindings
declare the `V_out` rows. Both may use realization records, incoming virtual
syndromes, and constants. Detector rows are intended to vanish on ideal
outcomes; the declared family may be redundant and is not assumed to be a
basis.

The gadget's affine outcome map has the separate form

```text
OM: M -> O
```

`LogicalObservable` declares one indexed action-outcome parity. Observable rows
use measurement records and constants, with no `V_in` terms. Random action
outcomes are valid observables too; they need not be deterministic detectors.
When a gadget has an `OutcomeMap` with `RESULT` rows, a `GadgetContract` derives
its observables from those rows. Explicit declarations must reproduce that
ordered affine table. The broader existing `OutcomeMap.input_syndromes` API
cannot be used in these observable rows.

A **gadget detector contract** consists of the detector map and the input and
output code presentations. `GadgetContract` is a compiler container that also
holds observable and success declarations; those additional fields do not
change the mathematical definition of a detector contract. Success rows are
acceptance mismatch parities; retry and selection policy stay in the protocol.

## Recipe-provided contracts

Surface-library gadgets expose their declared default contract through
`.contract`. Recipes also expose `gadget_contract(gadget)` functions:

```python
from cudaq.logical.architectures import surface

recipe = surface.definitions(3)
contract = recipe.memory_round.contract
assert contract is surface.gadget_contract(recipe.memory_round)
```

The Surface recipe returns cached contracts for the gadget objects it creates.
A gadget it did not create raises `ValueError`. For custom gadgets, declare a
`GadgetContract` explicitly or supply a default contract factory, as below.

The surface recipe supplies preparation and memory detectors, Clifford syndrome
transfers, and readout observables. Clifford transfers are derived from
physical gates, while measurement comparisons are selected by the recipe.

## Declare a contract

For a custom gadget, use `GadgetContractBuilder` to declare how recorded measurements
relate to its virtual boundary. This Steane memory round names its extracted
syndrome `round`:

```python
import cudaq.logical as cql


@cql.gadget(implements=cql.logical.idle)
def memory_round(
    block: cql.patch[cql.codes.Steane],
) -> cql.patch[cql.codes.Steane]:
    block, _ = cql.extract_syndrome(block, record="round")
    return block
```

`memory_round.record("round.s0")` refers symbolically to the first check result;
constructing the reference does not run the gadget. The check indices follow
the endpoint code profile's effective stabilizer ordering.

The incoming virtual syndrome contains the expected stabilizer signs inferred
from earlier results. In an ideal round, each current measurement agrees with
its incoming expectation, so their XOR forms a detector. The current results
then become the outgoing baseline:

```python
builder = cql.gadgets.GadgetContractBuilder(memory_round.interface)
incoming = memory_round.inputs.only().syndrome
outgoing = memory_round.outputs.only().syndrome
records = tuple(memory_round.record(f"round.s{i}") for i in range(incoming.width))

for i, record in enumerate(records):
    builder.detector(incoming[i] ^ record)
builder.bind(outgoing, records)
contract = builder.freeze()
```

`detector` declares a parity expected to vanish without errors. `bind` supplies
the complete outgoing virtual syndrome that a subsequent gadget can reference.
`freeze()` produces an immutable `GadgetContract`; these declarations add no
quantum operations or noise probabilities to the realization.

This memory round returns no logical measurement outcome, so it has no
observable rows. For a readout gadget, observables come from its declared
outcome map. Explicit observable declarations must agree with that map, as
explained in [Detector map and outcome map](#detector-map-and-outcome-map).

Compile the contract and verify the module:

```python
build = cql.compile(contract)
assert build.module.operation.verify()
```

Compilation materializes the authored contract declarations. Python
construction and MLIR verification check row structure, record ownership,
syndrome bounds, observable indices, and agreement with the outcome map.
[`analysis.edem_from_gadget`](../use-cases/gadget-edems.md#contract-checks-during-derivation)
additionally checks ideal determinism and completeness before constructing an
EDEM. It does not prove the declared logical action; use `verify_gadget` for
both checks.

Pass the explicit contract as an occurrence in a closed, compatible contract
sequence for
[`compiler.emit_stim`](../use-cases/stim-emission.md#save-the-stim-circuit). For
EDEM construction, pass the gadget and `contract=contract` to
[`analysis.edem_from_gadget`](../use-cases/gadget-edems.md#specify-faults):

```python
model = cql.analysis.edem_from_gadget(memory_round, contract=contract)
```

No separate compile step is required. Add `faults=` for an explicit fault set,
or `noise=` and optional `depolarizations=` for the standard noise policy.
An explicit fault set cannot be combined with the standard-noise arguments.
The contract is authored here; it is not inferred from the gadget definition.

### Declare a default contract

Use `@cql.gadget(contract=factory)` to attach a reusable contract declaration
to a custom gadget. The factory receives the gadget definition and returns
its `GadgetContract`:

```python
def memory_contract(gadget):
    incoming = gadget.inputs.only().syndrome
    outgoing = gadget.outputs.only().syndrome
    records = tuple(gadget.record(f"round.s{i}") for i in range(incoming.width))
    builder = cql.gadgets.GadgetContractBuilder(gadget.interface)
    for i, record in enumerate(records):
        builder.detector(incoming[i] ^ record)
    builder.bind(outgoing, records)
    return builder.freeze()


@cql.gadget(implements=cql.logical.idle, contract=memory_contract)
def declared_memory_round(
    block: cql.patch[cql.codes.Steane],
) -> cql.patch[cql.codes.Steane]:
    block, _ = cql.extract_syndrome(block, record="round")
    return block


model = cql.analysis.edem_from_gadget(declared_memory_round)
```

`declared_memory_round.contract` exposes the default declaration. An explicit
`contract=` argument selects a contract for that call. A gadget without a
declared default requires `contract=`; missing declarations raise an error
rather than triggering detector inference.

## Verify a contract and the logical action

The [authored-gadget example](../use-cases/examples/gadget-verification.md)
runs this workflow and demonstrates a failed logical-action receipt. The
[physical-memory example](../use-cases/examples/gadget-surface-memory.md)
applies it to recipe-provided preparation, extraction and readout contracts.

`verify_gadget` checks a `GadgetContract` against its gadget's realization:

```python
RepetitionPatch = cql.patch[cql.codes.Repetition]


@cql.gadget(implements=cql.logical.idle)
def syndrome_round(block: RepetitionPatch) -> RepetitionPatch:
    block, _ = cql.extract_syndrome(block)
    return block


block_in = syndrome_round.inputs.only()
block_out = syndrome_round.outputs.only()
checks = syndrome_round.records.syndrome(0).checks

contract = cql.gadgets.GadgetContract(
    syndrome_round,
    detectors=tuple(
        cql.gadgets.Detector(block_in.syndrome[i] ^ checks[i])
        for i in range(block_in.syndrome.width)),
    boundary={block_out.syndrome: checks},
)
receipt = cql.gadgets.verify_gadget(contract)
assert receipt.detector_contract.outcome_code_rank == 4
assert receipt.logical_action.action == "ql.logical.idle"
```

Both checks read one Choi probe of the realization. Every logical input is
Bell-paired with a reference qubit, the encoded input is placed in an
arbitrary stabilizer coset, and the realization's Clifford projection runs
through an outcome-complete symbolic stabilizer simulation over F2:

- **Detector contract.** Every declared detector and output virtual-syndrome
  relation must be a deterministic parity of the probed realization, and
  together with the deterministic combinations of the observables they must
  generate all of them (the rank condition). The receipt reports
  `outcome_code_rank == detector_rank + deterministic_logical_outcome_rank`.
  For the round above the Choi probe yields a 6 × 2 affine outcome relation
  with a four-dimensional left kernel, spanned by the two detectors and the
  two output relations: `4 = 4 + 0`.
- **Logical behavior.** The selected compiled gadget specification supplies
  the objective and its logical-operand bindings. Unitaries must reproduce
  every signed Choi relation. Preparations must have the declared logical
  eigenvalue. Measurements must correlate their mapped result with the
  corresponding reference Pauli and, for preserving measurements, with the
  surviving output Pauli. Untouched logical qubits must retain their identity
  Choi relations. `logical_action.verified_checks` counts the proved signed
  relations; `logical_rank` is its historical spelling, not a general rank.

A unitary may act on selected logical ports within a larger encoded boundary.
The remaining logical qubits must retain their identity action. Explicit
bindings select the operands; omitted bindings use canonical boundary order.

Failures raise `cudaq.logical.errors.GadgetVerificationError` and name the
condition; its `receipt` holds the full result. Unsupported claims also raise
by default. `verify_gadget(..., raise_on_failure=False)` returns failed or
unsupported receipts for inspection. Invalid declarations, compilation errors
and unsupported realization projections still raise diagnostics; inspection
receipts require a realization that can be analyzed. A normal return in the default
mode always has `receipt.verified == True`.
Omitting a detector fails the rank condition (`4 ≠ 3 + 0`). Declaring the same
round as `implements=cql.logical.x` passes the contract check, which is blind
to the logical action, and fails the action check with a sign mismatch on
`Z_ref -Z̄`; declaring `cql.logical.h` fails because `X_ref Z̄` does not
stabilize the Choi state.

The affine outcome map is part of the measurement meaning. Its constant is
checked along with its record terms: a uniformly random bit from an unrelated
ancilla is not an implementation of logical Z readout. Constants do not change
the separate rank/completeness calculation. Every supported check holds for
all input syndrome sectors and all ideal random outcomes, including hidden
reset outcomes.

Non-Clifford objectives (for example `cql.logical.t`), general instruments,
dynamic feed-forward and subsystem-code verification remain unsupported.
Measurement objectives contain one signed single-logical X/Z measurement,
with optional affine result processing and either full preservation or full
discard of the input logicals. Gates surrounding that ideal measurement,
multiple ideal measurements and partial quantum discard remain unsupported.
Classical-only objectives cannot pass through an empty set of quantum checks.
Inspection receipts report `logical_action.status == "unsupported"`, a
`reason`, and `receipt.verified == False`. The checks establish ideal
determinism and the declared supported logical behavior. They do not prove
fault tolerance or noisy circuit distance.

## Project the realization

`cudaq.logical.compiler.project_clifford` returns typed carrier instructions
and the existing `CompiledInterfaceManifest`, without parsing Stim text:

```python
projection = cql.compiler.project_clifford(build, gadget="memory_round")
instructions = projection.instructions
measurements = projection.interface.measurements
record_index = projection.record_indices["round.s0"]
assert measurements[record_index].index == record_index
```

For a gadget-root Build, `gadget=` defaults to its root symbol. Other roots
require an explicit gadget symbol. The selected gadget must have a linked
`GadgetSpec`, and every measurement needs a stable record name.

Each instruction has a kind, carrier operands or Pauli factors, a qualified
site ID, and a measurement start index when applicable. Calls and static
repeats are expanded with distinct invocation paths. Site IDs are stable for
a fixed realization; compiler rewrites may change them. Record aliases, such
as assembled syndrome rows, refer to existing measurement indices and do not
add measurements. Ambiguous aliases and record names beginning with the
reserved `__qlx_` prefix are rejected. Input and output encoded ports are
listed separately in `GadgetSpec` order, including their carrier partitions
and patch identities.
Preparation and destructive readout therefore retain their asymmetric
boundaries.

Currently, `fabric.permute` projects to ordinary `SWAP` instructions without
retaining that they represent a noiseless relabeling. This limits downstream
noise policies; see the
[permutation limitation](../use-cases/stim-emission.md#customize-the-experiment).

Projection verifies the module and rejects unsupported operations, recursive
calls, non-Clifford gates, and dynamic control. It preserves the source Build
and shares the native traversal used for Stim emission. The result describes
the ideal realization; projection does not insert noise, propagate faults,
produce an EDEM, or prove detector-contract correctness.

## Boundary for EDEM producers

Projection stays independent of EDEM storage and ownership. A subsequent
Logical analysis owns code algebra, contract verification and fault propagation,
then submits coefficients through EDEM's semantic construction API. Producers
name endpoints such as output syndrome of code block 2; they do not assemble
GF(2) type tables or packed offsets. EDEM validates the model and returns an
owned packed buffer. GF(2) supplies the checked map view and allocation-free
apply operation.

The producer integration uses the standalone `cudaq-edem` distribution as its
build and runtime dependency. That wheel supplies `cudaq_edem`, the native
libraries, headers and CMake exports; Logical uses that shared copy rather than
bundling another. cudaq-qec can depend on the same distribution.

The native producer hands its caller-held ownership record to Python through
the capsule protocol in `cudaq/edem/python.h`. `cudaq_edem.EDEM.adopt` validates
the packed bytes before consuming ownership, keeps the producer's lifetime
dependency alive, and retains the original payload address. Serialized input
can still use `EDEM.from_bytes`, which copies into aligned storage. Native
checked views and C++ objects do not cross this boundary.
