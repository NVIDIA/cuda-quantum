/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include <ostream>
#include <unordered_map>
#include <vector>

namespace cudaq {
/// @brief The resources type type encodes information regarding
/// the currently executing kernel's resource usage. This includes
/// number and type of quantum operation, circuit depth, etc.
class resources {
public:
  struct Instruction;

private:
  /// @brief We want an unordered_map on Instructions, so define
  /// the hash function here
  struct InstructionHash {
    std::size_t operator()(const Instruction &instruction) const;
  };

public:
  /// @brief The resources::Instruction is a data type that
  /// encapsulates the name of a quantum operation, the set of
  /// optional control indices, and the target qubit index.
  struct Instruction {
    /// @brief The name of the quantum instruction
    std::string name;

    /// @brief The optional set of control qubit indices
    std::vector<std::size_t> controls;

    /// @brief The target qubit index
    std::size_t target;

    /// @brief The constructor
    Instruction(std::string n, std::vector<std::size_t> c, std::size_t t)
        : name(n), controls(c), target(t) {}

    /// @brief Return true if this Instruction is equal to the given one.
    bool operator==(const Instruction &other) const;
  };

  resources() = default;
  resources(resources &) = default;
  resources(resources &&) = default;

  /// @brief Return the number of times the given Instruction is
  /// used in the current kernel execution
  std::size_t count(const Instruction &instruction) const;

  /// @brief Return the number of instructions with the given name and number of
  /// control qubits.
  std::size_t count(const std::string &name, std::size_t nControls) const;

  /// @brief Return the number of instructions with the given name
  std::size_t count(const std::string &name) const;

  /// @brief Append the given instruction to the resource estimate.
  void appendInstruction(const Instruction &instruction);

  /// @brief Dump resource count to the given output stream
  void dump(std::ostream &os) const;
  void dump() const;

private:
  /// @brief Map of Instructions in the current kernel to the
  /// number of times the Instruction is used.
  std::unordered_map<Instruction, std::size_t, InstructionHash> instructions;
};

} // namespace cudaq