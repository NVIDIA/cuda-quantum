/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/circuit/circuit.h"

#include <unordered_map>

namespace cudaq::gadgets {
inline namespace v0 {

/// Single-qubit Pauli channel: X, Y, or Z with these probabilities, else I.
struct PauliNoise {
  double x = 0;
  double y = 0;
  double z = 0;
};

/// Noise on each wire of a circuit, applied where the wire is defined;
/// classical wires have X (flip) noise only, and unlisted wires none. Output
/// noise repeats each output port's wire noise: qubits first, then bits.
struct CircuitNoise {
  /// Throws std::invalid_argument if num_q_out > output_noise.size().
  CircuitNoise(std::unordered_map<WireId, PauliNoise> noise,
               std::vector<PauliNoise> output_noise, std::size_t num_q_out)
      : noise(std::move(noise)) {
    if (num_q_out > output_noise.size())
      throw std::invalid_argument("more qubit outputs than output noise");
    q_output_noise.assign(output_noise.begin(),
                          output_noise.begin() + num_q_out);
    c_output_noise.assign(output_noise.begin() + num_q_out, output_noise.end());
  }

  PauliNoise get_noise(const WireId &wire) const {
    auto it = noise.find(wire);
    return it == noise.end() ? PauliNoise{} : it->second;
  }
  /// Flip probability of a classical wire.
  double get_bit_noise(const WireId &wire) const { return get_noise(wire).x; }

  /// Throws std::out_of_range for a bad port.
  PauliNoise get_output_noise(std::size_t port) const {
    return q_output_noise.at(port);
  }
  /// Flip probability of a classical output; throws std::out_of_range for a
  /// bad port.
  double get_output_bit_noise(std::size_t port) const {
    return c_output_noise.at(port).x;
  }

private:
  std::unordered_map<WireId, PauliNoise> noise;
  std::vector<PauliNoise> q_output_noise;
  std::vector<PauliNoise> c_output_noise;
};

struct NoiseModel {
  virtual ~NoiseModel() = default;
  virtual CircuitNoise noise(const InstructionCircuit &circuit) const = 0;
};

/// Noise per instruction, one PauliNoise per output (qubits, then bits),
/// applied at every call. Unlisted instructions are noiseless.
struct InstructionSetNoiseModel : NoiseModel {
  using InstructionNoise =
      std::unordered_map<std::string, std::vector<PauliNoise>>;

  /// Throws std::invalid_argument for an unknown name, a wrong entry count, or
  /// Y/Z noise on a classical output.
  InstructionSetNoiseModel(const PhysicalInstructionSet &instruction_set,
                           InstructionNoise noise);

  /// Throws std::invalid_argument if circuit does not use instruction_set.
  CircuitNoise noise(const InstructionCircuit &circuit) const override;

private:
  const PhysicalInstructionSet *instruction_set;
  InstructionNoise instruction_noise;
};

} // namespace v0
} // namespace cudaq::gadgets
