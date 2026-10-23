# Detector contracts and Clifford projection

A gadget has a logical action, a realization, encoded input and output
boundaries, and an affine outcome map. The existing `implements=` API calls the
action its **objective**. A detector contract describes how realization
measurements and virtual boundary syndromes determine detector parities and
output virtual syndromes.

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

The existing profile spellings `endpoint.syndrome`, `InputSyndromeRef`, and
`OutputSyndromeAssignment` describe **virtual syndromes**. They do not declare
residual-error coordinates. `CodeProfile` remains the compiler's larger code
algebra container; its selected generators and boundary coordinates supply a
code presentation. Profile syndrome bundles currently use the endpoint's
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
When a gadget has an `OutcomeMap` with `RESULT` rows, a `GadgetProfile` derives
its observables from those rows. Explicit declarations must reproduce that
ordered affine table. The broader existing `OutcomeMap.input_syndromes` API
cannot be used in these observable rows.

A **gadget detector contract** consists of the detector map and the input and
output code presentations. `GadgetProfile` is a compiler container that also
holds observable and success declarations; those additional fields do not
change the mathematical definition of a detector contract. Success rows are
acceptance mismatch parities; retry and selection policy stay in the protocol.

## Declare a contract

The profile graph can relate recorded syndrome-extraction results to a
gadget's virtual boundary:

```python
import cudaq.logical as cql


@cql.gadget(implements=cql.logical.idle)
def memory_round(
    block: cql.patch[cql.codes.Steane],
) -> cql.patch[cql.codes.Steane]:
    block, _ = cql.extract_syndrome(block, record="round")
    return block


graph = cql.gadgets.ProfileGraph(memory_round.interface)
incoming = memory_round.interface.inputs.only().syndrome
outgoing = memory_round.interface.outputs.only().syndrome
records = tuple(memory_round.record(f"round.s{i}") for i in range(incoming.width))

for i, record in enumerate(records):
    graph.detector(incoming[i] ^ record)
graph.bind(outgoing, records)
profile = graph.freeze()
build = cql.compile(profile)
assert build.module.operation.verify()
```

Python construction and MLIR verification check row structure, exact record
ownership, virtual-syndrome bounds, contiguous observable indices, and agreement
with a declared outcome map. They do **not** prove ideal determinism or
completeness. In particular, `boundary_complete` checks assignment coverage; it
does not establish that detector parities vanish on every ideal outcome or
that the contract captures all required relations.

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
listed separately in GadgetSpec order, including their carrier partitions
and patch identities.
Preparation and destructive readout therefore retain their asymmetric
boundaries.

Currently, `fabric.permute` projects to ordinary `SWAP` instructions without
retaining that they represent a noiseless relabeling. This limits downstream
noise policies; see the
[permutation limitation](../use-cases/stim-emission.md#export-a-profiled-surface-code-memory).

Projection verifies the module and rejects unsupported operations, recursive
calls, non-Clifford gates, and dynamic control. It preserves the source Build
and shares the native traversal used for Stim emission. The result describes
the ideal realization; projection does not insert noise, propagate faults,
produce an EDEM, or prove detector-contract correctness.
