/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "NoiseExtractor.h"
#include <algorithm>
#include <sstream>

namespace cudaq::ptsbe {

/// @brief Context information about an instruction being processed
struct InstructionContext {
  std::string name;
  std::size_t index;
  std::vector<std::size_t> target_qubits;
  std::vector<std::size_t> control_qubits;
};

/// @brief Extract target and control qubits from a trace instruction
///
/// @param inst Trace instruction
/// @return InstructionContext with extracted qubit information
static InstructionContext
extractInstructionContext(const cudaq::Trace::Instruction &inst,
                          std::size_t instruction_idx) {
  InstructionContext ctx;
  ctx.name = inst.name;
  ctx.index = instruction_idx;

  // Extract target qubits
  ctx.target_qubits.reserve(inst.targets.size());
  for (const auto &q : inst.targets) {
    ctx.target_qubits.push_back(q.id);
  }

  // Extract control qubits
  ctx.control_qubits.reserve(inst.controls.size());
  for (const auto &q : inst.controls) {
    ctx.control_qubits.push_back(q.id);
  }

  return ctx;
}

/// @brief Convert Kraus operators from cudaq::complex to std::complex<double>
///
/// @param channel Noise channel with Kraus operators
/// @return Vector of Kraus operator matrices in std::complex<double> format
static std::vector<std::vector<std::complex<double>>>
convertKrausOperators(const cudaq::kraus_channel &channel) {
  std::vector<std::vector<std::complex<double>>> operators;
  operators.reserve(channel.size());

  auto ops = channel.get_ops();
  for (const auto &kraus_op : ops) {
    std::vector<std::complex<double>> matrix;
    matrix.reserve(kraus_op.data.size());

    for (const auto &elem : kraus_op.data) {
      matrix.emplace_back(elem.real(), elem.imag());
    }

    operators.push_back(std::move(matrix));
  }

  return operators;
}

/// @brief Compute probabilities and normalized operators for a unitary mixture
///
/// Uses computeUnitaryMixture to validate and normalize Kraus operators.
/// Returns nullopt if the channel is not a valid unitary mixture.
///
/// @param kraus_operators Raw Kraus operator matrices
/// @param tolerance Numerical tolerance for validation
/// @return Pair of (probabilities, normalized_operators) or nullopt
static std::optional<
    std::pair<std::vector<double>, std::vector<std::vector<std::complex<double>>>>>
computeUnitaryMixtureData(
    const std::vector<std::vector<std::complex<double>>> &kraus_operators,
    double tolerance) {
  auto result = cudaq::computeUnitaryMixture(kraus_operators, tolerance);

  if (!result.has_value()) {
    return std::nullopt;
  }

  return result;
}

/// @brief Create a NoisePoint from a noise channel and instruction context
///
/// @param channel Noise channel to convert
/// @param ctx Instruction context (location, qubits, gate name)
/// @param tolerance Numerical tolerance for validation
/// @return Pair of (NoisePoint, is_valid_unitary_mixture)
static std::pair<NoisePoint, bool>
createNoisePointFromChannel(const cudaq::kraus_channel &channel,
                            const InstructionContext &ctx, double tolerance) {
  NoisePoint noise_point;
  noise_point.circuit_location = ctx.index;
  noise_point.op_name = ctx.name;

  noise_point.qubits = ctx.target_qubits;
  noise_point.qubits.insert(noise_point.qubits.end(),
                            ctx.control_qubits.begin(),
                            ctx.control_qubits.end());

  bool is_valid_unitary_mixture = true;

  // Check if channel has pre-computed unitary mixture representation
  if (channel.is_unitary_mixture()) {
    // Use pre-computed values (already validated)
    noise_point.kraus_operators = channel.unitary_ops;
    noise_point.probabilities = channel.probabilities;
  } else {
    // Convert Kraus operators and compute unitary mixture
    noise_point.kraus_operators = convertKrausOperators(channel);

    auto unitary_result =
        computeUnitaryMixtureData(noise_point.kraus_operators, tolerance);

    if (unitary_result.has_value()) {
      // Valid unitary mixture - use computed/normalized values
      noise_point.probabilities = std::move(unitary_result->first);
      noise_point.kraus_operators = std::move(unitary_result->second);
    } else {
      // Not a valid unitary mixture
      is_valid_unitary_mixture = false;

      // Assign uniform probabilities as fallback (for non-validating mode)
      noise_point.probabilities.resize(
          noise_point.kraus_operators.size(),
          1.0 / noise_point.kraus_operators.size());
    }
  }

  return {std::move(noise_point), is_valid_unitary_mixture};
}

/// @brief Validate a NoisePoint and throw if validation is enabled
///
/// @param noise_point NoisePoint to validate
/// @param ctx Instruction context (for error messages)
/// @param validate_unitary_mixture Whether to throw on validation failure
/// @param tolerance Numerical tolerance
/// @return true if valid, false otherwise
static bool validateNoisePoint(const NoisePoint &noise_point,
                               const InstructionContext &ctx,
                               bool validate_unitary_mixture,
                               double tolerance) {
  if (!noise_point.isUnitaryMixture(tolerance)) {
    if (validate_unitary_mixture) {
      std::ostringstream msg;
      msg << "Noise channel validation failed for gate '" << ctx.name
          << "' at instruction " << ctx.index
          << ". Channel does not satisfy unitary mixture properties.";
      throw std::invalid_argument(msg.str());
    }
    return false;
  }
  return true;
}

/// @brief Throw validation error for non-unitary mixture channel
///
/// @param ctx Instruction context (for error message)
static void throwUnitaryMixtureError(const InstructionContext &ctx) {
  std::ostringstream msg;
  msg << "Noise channel for gate '" << ctx.name << "' at instruction "
      << ctx.index << " is not a valid unitary mixture. "
      << "PTSBE requires all channels to be unitary mixtures.";
  throw std::invalid_argument(msg.str());
}

NoiseExtractionResult
extractNoiseSites(const cudaq::Trace &trace,
                  const cudaq::noise_model &noise_model,
                  bool validate_unitary_mixture, double tolerance) {

  NoiseExtractionResult result;
  result.total_instructions = trace.getNumInstructions();
  result.noisy_instructions = 0;
  result.all_unitary_mixtures = true;

  std::size_t instruction_idx = 0;

  // Iterate through all instructions in circuit execution order
  for (const auto &inst : trace) {
    // Extract instruction context (qubits, name, index)
    auto ctx = extractInstructionContext(inst, instruction_idx);

    // Query noise model for applicable channels
    auto channels = noise_model.get_channels(ctx.name, ctx.target_qubits,
                                             ctx.control_qubits, inst.params);

    // Process each applicable noise channel
    for (const auto &channel : channels) {
      // Skip empty channels
      if (channel.empty()) {
        continue;
      }

      // Create NoisePoint from channel
      auto [noise_point, is_valid_unitary] =
          createNoisePointFromChannel(channel, ctx, tolerance);

      // Track unitary mixture status
      if (!is_valid_unitary) {
        result.all_unitary_mixtures = false;
        if (validate_unitary_mixture) {
          throwUnitaryMixtureError(ctx);
        }
      }

      // Validation of the constructed NoisePoint
      bool point_valid =
          validateNoisePoint(noise_point, ctx, validate_unitary_mixture, tolerance);
      if (!point_valid) {
        result.all_unitary_mixtures = false;
      }

      result.noise_sites.push_back(std::move(noise_point));
      result.noisy_instructions++;
    }

    instruction_idx++;
  }

  return result;
}

} // namespace cudaq::ptsbe
