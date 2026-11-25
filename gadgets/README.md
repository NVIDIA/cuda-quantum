# cudaq-gadgets

## Layout

Each component depends only on those above it.

| Component | Headers | Sources | Contents |
| --- | --- | --- | --- |
| core | [core.h](include/cudaq/gadgets/core.h) | [core.cpp](lib/core.cpp) | C ABI version |
| utils | [utils.h](include/cudaq/gadgets/utils.h) | [utils.cpp](lib/utils.cpp) | Paulis and operation payloads |
| circuit | [circuit/](include/cudaq/gadgets/circuit/) `circuit.h` `standard.h` `instructions.h` | [circuit/](lib/circuit/) | Base nodes and circuits; semantic operations and circuits; instruction circuits and sets |
| gadget | [gadget/gadget.h](include/cudaq/gadgets/gadget/gadget.h) | — | Codes, gadgets, profiles, gadget circuits |
| noise | [noise/noise.h](include/cudaq/gadgets/noise/noise.h) | [noise/](lib/noise/) | Pauli noise and noise models over circuits |
| analysis | [analysis/analysis.h](include/cudaq/gadgets/analysis/analysis.h) | [analysis/](lib/analysis/) | Verification and EDEM generation |

Everything builds into one library, `cudaq-gadgets`. Tests are in [test/](test/).

## Where things are defined

| Structure | Definition |
| --- | --- |
| `Pauli`, `PauliString`, `Shape`, `Wires`, `CliffordImages`, `ConditionalPauli`, `DiagonalUnitary`, `Unitary` | [utils.h](include/cudaq/gadgets/utils.h) |
| `Opcode`, `Payload`, `get_shape`, `to_string`, `find_opcode`, `OpcodeNode`, `SemanticCircuit` | [circuit/standard.h](include/cudaq/gadgets/circuit/standard.h); table in [standard.cpp](lib/circuit/standard.cpp) |
| `Node`, `Circuit`, `DebugInfo` | [circuit/circuit.h](include/cudaq/gadgets/circuit/circuit.h) |
| `InstructionNode`, `InstructionCircuit`, `InstructionSet`, `PhysicalInstructionSet` | [circuit/instructions.h](include/cudaq/gadgets/circuit/instructions.h) |
| `PhysicalInstructionDefinition` | [circuit/instructions.h](include/cudaq/gadgets/circuit/instructions.h); signature tracing in [instructions.cpp](lib/circuit/instructions.cpp) |
| `SubsystemCode`, `GadgetProfile`, `BlockBinding`, `OutcomeMap` | [gadget/gadget.h](include/cudaq/gadgets/gadget/gadget.h) |
| `Gadget`, `GadgetSet` | [gadget/gadget.h](include/cudaq/gadgets/gadget/gadget.h) |
| `PauliNoise`, `CircuitNoise`, `NoiseModel`, `InstructionSetNoiseModel` | [noise/noise.h](include/cudaq/gadgets/noise/noise.h) |
| `EDEM`, `VerificationReport`, `verify`, `get_edem` | [analysis/analysis.h](include/cudaq/gadgets/analysis/analysis.h) |

## Differences from the spec

- Wires are single-assignment: each wire ID is defined once and names one
  segment.
- There are no subroutine or repeat nodes.
- A `SemanticCircuit` holds semantic operations (`Opcode`, with payloads). An
  `InstructionCircuit` calls the instructions of a physical instruction set by
  index.
- Gadget circuits are not supported yet.
- Noise is separate from gadgets and instruction sets. A `NoiseModel` maps a
  circuit to per-wire Pauli noise. A standard definition for uniform noise over
  instruction sets is provided. Otherwise a custom noise model can specify noise
  per wire based on the entirety of the circuit however it wants.

## TODO

Construction and validation are TODOs throughout: structures can be built,
but almost nothing checks them, and the spec's algorithms (verification, EDEM
generation, composition) are stubs or missing. `TODO` comments in the headers
mark what each structure still needs.

## Build and test

In-tree, the library builds with CUDA-Q:

```bash
bash scripts/build_cudaq.sh -c Debug -i
build/gadgets/test/gadgets_core_test
build/gadgets/test/gadgets_circuit_test
build/gadgets/test/gadgets_gadget_test
```
