# Derive a gadget EDEM

The [Stim emission tutorial](stim-emission.md#export-a-profiled-surface-code-memory)
exports a circuit containing quantum operations, noise, detectors, and a logical
observable. A **detector error model (DEM)** describes how faults in that circuit
flip detectors and logical observables, together with their probabilities.
Stim can derive a DEM by analyzing the complete circuit.

An **extended detector error model (EDEM)** describes these effects for an
individual gadget while keeping its encoded inputs and outputs open. It also
tracks error information entering and leaving those boundaries. This lets you
derive a model for each gadget, reuse it for repeated occurrences, and compose
the models into a larger experiment. Once preparation and readout close all
encoded boundaries, the composed model can be projected to a DEM.

`cudaq.logical.analysis.derive_edem` builds the gadget models;
`cudaq_edem` provides composition and DEM operations. The example below
demonstrates this modular construction and checks its result against Stim's
analysis of the complete circuit. Circuit export supplies the reference for
that check; the EDEM path derives its result from the gadget models.

## Compare circuit and per-gadget DEM construction

The [memory comparison example](../../../examples/standalone/07_memory_edem.py)
uses the same built-in surface recipe and experiment as the Stim emission
tutorial. It computes the DEM in two ways:

1. Emit the complete Stim circuit and obtain its DEM with
   `circuit.detector_error_model()`.
2. Derive EDEMs for preparation, a memory round, and readout; compose their
   occurrences; then project the result to a DEM and convert it to the same
   independent-error form for comparison.

Both paths use the same detector profiles and noise placements. Their agreement
checks that the modular construction reproduces the complete-circuit model.
The file's `# %%` cells let you inspect each stage interactively.

Use an environment with CUDA-Q Logical installed; its wheel also installs the
required `cudaq-edem` dependency. This example additionally requires Stim for
the comparison:

```bash
python -m pip install 'stim>=1.16,<2'
```

### Import the libraries

Import Logical, the surface recipe and geometry, the EDEM operations, and Stim:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: from math import isclose
   :end-before: # %%
```

The script's `detector_profile` and `data_noise_placements` helpers are explained
in [Define detector profiles](stim-emission.md#define-detector-profiles) and
[Define data-noise placements](stim-emission.md#define-data-noise-placements).
The walkthrough below continues with the experiment setup and EDEM construction.

### Configure and assemble the experiment

Keep the same distance, round count, and data-error probability as the circuit
export example. The additional `output` setting controls whether to save the
results; `None` keeps them in memory:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: distance = 3
   :end-before: # %%
```

Select the built-in surface definitions and data-qubit geometry:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: implementation = surface.definitions(distance)
   :end-before: # %%
```

Use the shared helpers to build the profile sequence and its noise placements:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: prep = detector_profile(
   :end-before: # %%
```

At the defaults, the sequence has four occurrences: preparation, two memory
rounds, and logical-Z readout. The two memory rounds reuse the same profile.

### Obtain the circuit-derived reference DEM

Export the complete circuit using the same Python exporter as the Stim emission
tutorial:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: coordinates =
   :end-before: # %%
```

Ask Stim to derive a DEM from that circuit's operations, annotations, and noise:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: circuit = stim.Circuit(text)
   :end-before: # %%
```

Keep `circuit_dem` as the reference. The following steps construct the other
DEM from the individual gadgets.

### Derive reusable gadget EDEMs

For each profile, compile its gadget and obtain the physical Clifford
instructions with `project_clifford`. Then construct a **fault set**, the
possible physical faults and their probabilities, and derive the EDEM:

| Operation | Purpose |
| --- | --- |
| `cql.compile(profile)` | Build the gadget with its detector profile. |
| `project_clifford(build, ...)` | Expose the physical instructions and fault insertion sites. |
| `standard_depolarizing_faults(...)` | Construct faults using the same noise policy and placements as circuit export. |
| `derive_edem(build, faults=faults)` | Compute fault effects and error transfer through the gadget's encoded boundaries. |

The cache uses a profile and its noise placements as the key. With the default
parameters, it derives three models: preparation, one memory round, and readout.
`occurrences` lists the four models in execution order, reusing the memory-round
model for both rounds:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: cache, occurrences =
   :end-before: # %%
```

### Compose the models and close the boundaries

`compose_chain` connects each occurrence's encoded outputs to the next one's
inputs, carrying error information forward to later detectors and readout.
Repeated occurrences receive separate fault indicators, so reusing a model
does not correlate the faults in different rounds.

Preparation starts the patch and readout consumes it. The complete chain
therefore has no open encoded inputs or outputs, as the assertion checks.
`project_DEM` converts that closed EDEM to a DEM while preserving its fault
probabilities and mutually exclusive alternatives:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: composed = edem.compose_chain
   :end-before: # %%
```

### Put the DEM in independent-error form

`projected` is the composed DEM. To compare it with Stim's DEM, the example
converts its fault laws to the independent-error form used by Stim.

A depolarizing fault chooses mutually exclusive Pauli alternatives: X, Y, or Z
on one qubit. `factorize_dem` converts supported depolarizing laws to equivalent
independent mechanisms without changing the resulting error distribution.
`coalesce_dem_columns` then merges mechanisms with the same detector and
observable flips, combining their probabilities according to whether an odd
number of those mechanisms occur.

The local `to_stim_dem` helper formats the resulting mechanisms as a
`stim.DetectorErrorModel` for the comparison:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: independent = edem.factorize_dem(
   :end-before: # %%
```

`source_columns` records where each column goes during coalescing, with `None`
for discarded columns. The original mutually exclusive fault laws remain
available in `projected`; `to_stim_dem` only formats the transformed model.

### Compare the two results

Check that both DEMs have the same detector and observable counts. The local
`dem_errors` helper reads each error term into a dictionary keyed by the
detectors and observables it flips. Comparing the keys checks that both models
contain the same error patterns; comparing the values checks their
probabilities with an absolute tolerance of `1e-12`:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: assert (circuit_dem.num_detectors
   :end-before: # %%
```

Detector coordinates and timing annotations are excluded from this comparison.
`dem_errors` expects coalesced terms and rejects duplicate error patterns.

### Run the example and save its results

Run from the repository root:

```bash
python preview/logical/examples/standalone/07_memory_edem.py
```

With the default parameters, it reports agreement for 20 detectors, one logical
observable, and 53 error terms. Four occurrences use three derived models and
retain 81 original fault indicators.

Set `output = Path("memory-comparison")` in the parameter cell to save the
circuit, both Stim DEMs, per-definition EDEMs, the composed EDEM, and the
projected packed DEM. The final cell performs the export:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/07_memory_edem.py
   :language: python
   :start-at: if output is not None:
```

### Model and comparison constraints

EDEM composition currently supports serial chains, and adjacent encoded
boundaries must use matching code presentations. DEM projection requires all
encoded boundaries to be closed; it preserves the original fault laws.

The independent-error conversion supports uniform one- and two-qubit
depolarization with total probabilities at most `3/4` and `15/16`, respectively.
Other exclusive laws return an unsupported-distribution error from
`factorize_dem`, although they can still be retained in the projected DEM.
Coalescing combines mechanisms with identical complete responses, including
any success bits, and removes zero responses and zero-probability mechanisms.

Postselection accepts only selected measurement outcomes. `project_DEM`
preserves these success outputs, but the example's Stim formatter rejects
nonempty success outputs because it has no postselection semantics.

## A boundary-transfer example

To inspect error transfer through an open boundary, consider a one-qubit H
gadget. H turns an incoming X error into a Z error. The example inserts an X
fault at the input and checks the resulting logical error:

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
The [detector-contract reference](../reference/detector-contracts.md#virtual-syndromes-and-boundary-errors)
defines the virtual-syndrome changes `ΔV`, residual-error syndrome `S`, and
logical-error coordinates `L` used at encoded boundaries.

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

<!--
% invisible-code-block: python
%
% profiles = load_ql_example(
% "preview/logical/examples/standalone/07_memory_edem.py", "profiles")
-->

```python
round_profile = profiles[1]
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
