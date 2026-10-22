# Derive a gadget EDEM

`cudaq.logical.analysis.derive_edem` derives an extended detector error model
from a gadget realization, its declared detector contract, and an explicit
finite set of faults. It returns a `cudaq_edem.EDEM` value with open encoded
boundaries. The [detector-contract reference](../reference/detector-contracts.md)
explains the distinction between virtual syndromes and boundary-error
coordinates.

EDEM is a required build and runtime dependency of Logical. Installing the
`cudaq-logical` wheel also installs `cudaq-edem`. Source builds use its CMake
package or the enclosing CUDA-Q build's EDEM target.
The standalone wheel supplies one shared native copy; Logical does not bundle
another EDEM library.

## Compare circuit and per-gadget DEM construction

The [memory comparison example](../../../examples/standalone/07_memory_edem.py)
uses the same surface-memory profiles, `StandardDepolarizingNoise` policy,
and explicit extra data-noise placements for two paths:

1. Emit the complete Stim circuit and obtain its DEM with
   `circuit.detector_error_model()`.
2. Derive a packed EDEM for each distinct gadget definition, compose the
   serial chain with `cudaq_edem.compose_chain`, and project the closed result
   with `cudaq_edem.project_DEM`. Call `cudaq_edem.factorize_dem`, then
   `cudaq_edem.coalesce_dem_columns`, before rendering the independent-error
   DEM with the example's optional `to_stim_dem` formatter.

`analysis.standard_depolarizing_faults` constructs the finite `FaultSet` using
the same standard-noise placement as circuit export. Measurement noise is a
physical Pauli immediately before measurement; it also affects later uses of
that carrier. Extra data-noise layers remain explicit recipe inputs, passed as
`compiler.Depolarize1Placement` values through the `depolarizations` argument
to both APIs. The memory recipe places them immediately before each extraction,
including the extractions inside preparation and final readout. They have a
separate `data_depolarization` probability, independent of the operation noise.

The native EDEM library connects the `(ΔV, S, L)` boundary wires and checks
adjacent code presentations. Each occurrence keeps its own fault indicators
and unmodified probability groups, including repeated uses of the same SEC
definition. `project_DEM` rejects exposed code boundaries and returns an owned
`cudaq_edem.DEM`; it preserves the original columns and exact fault laws. The
Stim adapter only renders the transformed columns and probabilities. All
factorization and column coalescing run in the native EDEM library. Stim remains
an example dependency.

<!--
% invisible-code-block: python
%
% compare = load_ql_example(
% "preview/logical/examples/standalone/07_memory_edem.py", "compare")
-->

```python
(memory, occurrences, composed, projected, circuit, circuit_dem,
 edem_dem) = compare(distance=3, rounds=5)
assert len(occurrences) == 5
assert occurrences[1] is occurrences[2]
assert composed.inputs == composed.outputs == []
assert sum(composed.detectors) == circuit.num_detectors == 40
assert sum(composed.observables) == 1

from cudaq_edem import factorize_dem, coalesce_dem_columns

independent = factorize_dem(projected)
reduced, source_to_result = coalesce_dem_columns(independent)
assert len(source_to_result) == projected.vector_sizes[0]
assert sum(reduced.detectors) == sum(projected.detectors)
```

`compare` checks the DEM dimensions and the probability of every distinct
detector/observable error support after native column coalescing, with an
absolute numerical tolerance of `1e-12`. Its comparison helper rejects duplicate
terms instead of performing another probability transformation. Detector coordinates
and timing metadata are excluded. Tests also compare against Stim's generated
rotated-memory-Z circuit at distances 3 and 5.

The memory recipe resets ancillas at the start of the extraction that uses
them. Reset faults therefore belong to that gadget, and only data-code errors
need to cross encoded boundaries. Moving these resets across gadget cuts
preserves their order on each physical carrier. The final unused ancilla
resets retain their fault indicators with zero response.

The packed EDEMs and projected DEM preserve exclusive Pauli alternatives.
The native `factorize_dem` operation performs exact independent-error
factorization for uniform one- and two-qubit depolarization; it does not
approximate exclusive alternatives as independent events with their original
probabilities. As with Stim's DEM analysis, the
supported total probabilities are at most `3/4` and `15/16`, respectively.
Other exclusive laws return an unsupported-distribution error from factorization.
DEM projection retains them without approximation. Nonempty success outputs
are likewise retained by `project_DEM` and rejected by the Stim adapter, which
has no postselection semantics.

`coalesce_dem_columns` then combines independent mechanisms with identical
complete responses, including observable and success bits, using the exact
parity probability. It removes zero responses and zero-probability combined
mechanisms. Global rows retain their numbering; output partitions consolidate
to one per kind. The returned mapping links each original response column to
its reduced column, or `None` when discarded. Factorization keeps the original
response column order, so the map can refer back to occurrence/local-fault
coordinates in `projected`. It is not a mapping of sampled independent events
back to the original categorical events; retain `projected` for the physical
fault laws. The formatter neither changes probabilities nor removes or combines
columns.

To save the circuit, both Stim DEMs, the per-definition EDEMs, the composed
EDEM, and the projected packed DEM:

```bash
python -m pip install 'stim>=1.16,<2'
python preview/logical/examples/standalone/07_memory_edem.py --distance 3 --rounds 3 --output memory-comparison
```

## A boundary-transfer example

```python
import cudaq.logical as cql
from cudaq.logical.analysis import FaultSet, derive_edem


@cql.gadget(implements=cql.logical.h)
def bare_h(block: cql.patch[cql.codes.BareQubit]) -> cql.patch[cql.codes.BareQubit]:
    return cql.h(block.data)


build = cql.compile(cql.gadgets.GadgetProfile(bare_h))
projection = cql.compiler.project_clifford(build, gadget="bare_h")

# An empty insertion site denotes the input boundary. H sends this X to Z.
faults = FaultSet().pauli_error("", 0, "x", 0.01)
model = derive_edem(build, faults=faults)
assert model.apply([1, 0, 0]) == [0, 1]

# Packed values use EDEM's checked reader and retain canonical bytes.
from cudaq_edem import EDEM
assert EDEM.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
```

The input vector contains the fault indicators followed by `(ΔV, S, L)` for
each input code block. The output contains detector, observable, and success
flips followed by `(ΔV, S, L)` for each output code block. `L` uses interleaved
logical X and Z coordinates. This bare code has no stabilizers, so both `ΔV`
and `S` have width zero. Input and output code-block counts are independent;
preparation and destructive readout retain their asymmetric boundaries.

`model.port(Endpoint(PortKind.LOGICAL), output=True)` returns the output
logical width and offset. Other metadata and matrix inspection use the
`cudaq_edem` interface. Logical supplies semantic blocks to the native EDEM
constructor; EDEM owns layout, validation, packing, and evaluation.

`derive_edem` transfers the native constructor's owned packed buffer directly
into `cudaq_edem.EDEM` through validated adoption. The hand-off preserves the
payload address and retains the producer's lifetime dependency until release.
It avoids intermediate string and Python-byte copies. `to_bytes` and
`from_bytes` remain explicit serialization conveniences. See the
[producer boundary](../reference/detector-contracts.md#boundary-for-edem-producers)
for the ownership contract.

## Declare and check the detector contract

For a recorded syndrome-extraction gadget, declare detector parities and
output virtual-syndrome bindings using `ProfileGraph`, as shown in
[Declare a contract](../reference/detector-contracts.md#declare-a-contract).
Compile that profile and pass the Build to `derive_edem`.

Derivation proves the following ideal conditions using an exact affine
Clifford tableau with encoded logical inputs paired to inaccessible Choi
references and independent input stabilizer signs:

- Every declared detector is identically zero, including its affine constant.
- Every output virtual-syndrome formula agrees with the actual ideal output
  stabilizer sign, including signs introduced by products of Pauli operators.
- Success mismatch parities agree with their accepted ideal value.
- The declared detectors, output relations, success rows, and deterministic
  combinations of action outcomes span all deterministic outcome relations.

Detector families may be redundant and may include zero rows. Observables may
be random; only their deterministic combinations contribute to the
completeness rank. `boundary_complete` alone establishes assignment coverage,
so a structurally complete profile can still fail this semantic check.

These checks establish detector-contract correctness. They do not certify that
the realization implements its declared logical action. Action verification
remains a separate analysis.

For redundant stabilizer presentations, Logical constructs the input section
using the profile's `kept_from_effective` matrix and the code's canonical
anti-stabilizers. This is a section on attainable syndrome coordinates.
Evaluation on an inconsistent redundant bitstring does not give that bitstring
a physical interpretation. Presentation type names encode a versioned digest
of the ordered boundary algebra, so equal dimensions do not alias different
generators or logical representatives. These names do not constitute an EDEM
registry or an exchange-envelope signature.

## Specify faults

Insertion sites come from `projection.instructions[i].site_id`. Insertions
occur immediately after that instruction and must target a live carrier.
Site IDs belong to a fixed realization; obtain a new projection after circuit
rewrites. A stable record flip changes the classical measurement bit without
adding a physical Pauli error:

```python
round_profile = memory.profiles[1]
round_build = cql.compile(round_profile)
round_projection = cql.compiler.project_clifford(
    round_build, gadget=round_profile.gadget.name)
last_check_site = round_projection.instructions[-1].site_id
data_carrier = round_projection.interface.inputs[0].data_carriers[0]
faults = FaultSet().record_flip(round_projection.interface.measurements[0].record, 0.01)
faults = faults.pauli_error(last_check_site, data_carrier, "x", 0.02)
model = derive_edem(round_build, faults=faults)
```

Record paths and non-measuring aliases resolve to exact physical measurement
indices. A faulty measurement can change outgoing `ΔV` without changing
outgoing `S`; a late data fault can change `S` without changing `ΔV`.
Incoming `ΔV` contributes through the detector map, with structural zeros to
observable flips and outgoing `(S, L)`.

`ElementaryFault` can group several `PauliInsertion` values and record flips
into one correlated indicator. `FaultSet.pauli_channel`, `depolarize1`, and
`depolarize2` retain categorical alternatives as mutually exclusive original
indicators. Ungrouped faults are independent. Equal effect columns keep their
separate indicators and probabilities.
Fault column `i` corresponds to `faults.faults[i]`; retain that ordered input
set when interpreting source IDs, which are not stored in the packed EDEM.

Individual probabilities must be finite and within `[0, 1]`. EDEM performs the
exclusive-total check with its current library policy; Logical adds no
separate tolerance or renormalization. The same native constructor is used by
direct EDEM construction and this producer.

The first producer supports canonical stabilizer codes (`r = 0`), the typed
Clifford projection, static calls and repeats, Pauli insertions, and record
flips. The native producer requires an explicit fault set; the memory example
constructs one for its standard noise policy. Erasures, subsystem boundaries,
dynamic feedback, and non-Clifford realizations are outside this slice.
EDEM composition currently supports serial chains; general diagrams and
window construction are outside this slice. Unsupported inputs fail with diagnostics.

## Build from source

From the CUDA-Q repository root, build and install the EDEM dependency:

```sh
cmake -S edem -B build-edem -DCUDAQ_EDEM_BUILD_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=/path/to/edem-install
cmake --build build-edem
cmake --install build-edem
```

Then add this option to the normal
[standalone Logical configuration](../reference/building-against-cudaq.md):

```sh
-Dcudaq-edem_DIR=/path/to/edem-install/lib/cmake/cudaq-edem
```

Make the installed `cudaq_edem` Python package available to the same Python
interpreter. CMake also discovers an installed `cudaq-edem` wheel through that
interpreter, so no package path is needed when using the wheel. In a parent
build that already defines the `cudaq::edem` target, Logical uses that target
directly. Enabling Logical in the CUDA-Q source build also builds EDEM and
stages its Python package alongside Logical.
