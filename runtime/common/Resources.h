/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Trace.h"
#include <ostream>
#include <unordered_map>
#include <vector>

namespace cudaq {
/// @brief The Resources type encodes information regarding
/// the currently executing kernel's resource usage. This includes
/// number and type of quantum operation, circuit depth, etc.
class Resources {
public:
  struct Instruction;

private:
  /// @brief We want an unordered_map on Instructions, so define
  /// the hash function here
  struct InstructionHash {
    std::size_t operator()(const Instruction &instruction) const;
  };

public:
  /// @brief The Resources::Instruction is a data type that
  /// encapsulates the name of a quantum operation, the set of
  /// optional control indices, and the target qubit index.
  struct Instruction {
    /// @brief The name of the quantum instruction
    std::string name;

    /// @brief The number of controls
    std::size_t nControls;

    Instruction(const std::string &n) : name(n), nControls(0) {}

    /// @brief The constructor
    Instruction(const std::string &n, const size_t c) : name(n), nControls(c) {}

    /// @brief Return true if this Instruction is equal to the given one.
    bool operator==(const Instruction &other) const;
  };

  Resources() = default;
  Resources(Resources &) = default;
  Resources(Resources &&) = default;

  /// @brief Return the number of times the given Instruction is
  /// used in the current kernel execution
  std::size_t count(const Instruction &instruction) const;

  /// @brief Return the number of instructions with the given name and number of
  /// control qubits.
  std::size_t count_controls(const std::string &name,
                             std::size_t nControls) const;

  /// @brief Return the number of instructions with the given name
  std::size_t count(const std::string &name) const;

  /// @brief Return the total number of operations
  std::size_t count() const;

  /// @brief Append the instruction with the given name and number of controls
  /// to the resource estimate.
  void appendInstruction(const std::string &name, std::size_t nControls,
                         std::size_t count = 1);

  /// @brief Dump resource count to the given output stream
  void dump(std::ostream &os) const;
  void dump() const;

  /// @brief Clear the resource usage counts
  void clear();

  /// @brief Register the usage of an additional qubit
  void addQubit();

  /// @brief Returns a dictionary mapping gate names to counts
  std::unordered_map<std::string, std::size_t> gateCounts() const;

private:
  /// @brief Map of Instructions in the current kernel to the
  /// number of times the Instruction is used.
  std::unordered_map<Instruction, std::size_t, InstructionHash> instructions;

  /// @brief Keep track of the total number of gates. We keep this
  /// here so we don't have to keep recomputing it.
  std::size_t totalGates = 0;

  /// @brief Keep track of the total number of qubits used.
  std::size_t numQubits = 0;
};

} // namespace cudaq
