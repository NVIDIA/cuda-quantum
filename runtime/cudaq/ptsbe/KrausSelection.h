/**********************************-*- C++ -*-**********************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace cudaq {

/// @brief Represents the choice of a specific Kraus operator at one noise point
struct KrausSelection {
  /// @brief Unique position in the circuit's total ordering of noise-capable
  /// operations Each operation that can have noise applied is assigned a
  /// sequential position (0, 1, 2, ...)
  std::size_t circuit_location = 0;

  /// @brief Qubits affected by this noise operation (controls + targets)
  std::vector<std::size_t> qubits;

  /// @brief The gate operation after which this noise occurs (e.g., "h", "x",
  /// `"cx"`)
  std::string op_name;

  /// @brief Which Kraus operator from the noise channel was selected
  std::size_t kraus_operator_index = 0;

  /// @brief Whether this selection represents an actual error (non-identity).
  /// Set at trajectory generation time where the noise channel is available.
  bool is_error = false;

  /// @brief Default constructor
  KrausSelection() = default;

  /// @brief Constructor with all fields
  /// @param location Unique position in circuit's noise operation sequence
  /// @param qbits Qubits affected by this noise operation
  /// @param op Gate operation name (e.g., "h", "x", `"cx"`)
  /// @param idx Selected Kraus operator index from noise channel
  /// @param error Whether this selection is a non-identity error
  KrausSelection(std::size_t location, std::vector<std::size_t> qbits,
                 std::string op, std::size_t idx, bool error = false)
      : circuit_location(location), qubits(std::move(qbits)),
        op_name(std::move(op)), kraus_operator_index(idx), is_error(error) {}

  /// @brief Equality comparison for testing
  /// @param other KrausSelection to compare with
  /// @return true if all fields are equal
  constexpr bool operator==(const KrausSelection &other) const {
    return circuit_location == other.circuit_location &&
           qubits == other.qubits && op_name == other.op_name &&
           kraus_operator_index == other.kraus_operator_index &&
           is_error == other.is_error;
  }
};

} // namespace cudaq
