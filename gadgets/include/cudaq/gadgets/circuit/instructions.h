/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/circuit/standard.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudaq::gadgets {
inline namespace v0 {

// Pointers between objects are non-owning.
template <typename Instruction>
struct InstructionSet;
struct PhysicalInstructionDefinition;
using PhysicalInstructionSet = InstructionSet<PhysicalInstructionDefinition>;

/// A call to the instruction at an index of an instruction set.
struct InstructionNode : Node {
  InstructionNode(std::size_t instruction, Wires wires)
      : Node(std::move(wires)), instruction(instruction) {}
  std::size_t get_instruction() const { return instruction; }

private:
  std::size_t instruction;
};

/// A circuit of calls to the instructions of an instruction set.
struct InstructionCircuit : Circuit<InstructionNode> {
  explicit InstructionCircuit(const PhysicalInstructionSet &instruction_set,
                              Wires wires = {})
      : Circuit(std::move(wires)), instruction_set(&instruction_set) {}

  const PhysicalInstructionSet &get_instruction_set() const {
    return *instruction_set;
  }
  /// Appends node and returns its index. Throws
  /// std::invalid_argument if its instruction is not in the set, or its wires
  /// do not fit the instruction.
  std::size_t append(InstructionNode node) override;

  /// Appends an InstructionNode built from instruction and wires.
  std::size_t append(std::size_t instruction, Wires wires) {
    return append(InstructionNode(instruction, std::move(wires)));
  }

private:
  const PhysicalInstructionSet *instruction_set;
};

/// A physical instruction, defined by an action of standard operations whose
/// boundary wires bind to its ports by index.
struct PhysicalInstructionDefinition {
  /// Takes its shape from action's boundary, and derives port_map by tracing
  /// each input to the output it reaches.
  /// TODO: Queries for port flags (prepared, reset, discarded, inactive).
  PhysicalInstructionDefinition(std::string name, SemanticCircuit action);

  const std::string &get_name() const { return name; }
  const Shape &get_shape() const { return shape; }
  /// (input port, output port) for each qubit passing through.
  const std::vector<std::pair<std::size_t, std::size_t>> &get_port_map() const {
    return port_map;
  }
  const SemanticCircuit &get_action() const { return action; }

private:
  std::string name;
  Shape shape;
  std::vector<std::pair<std::size_t, std::size_t>> port_map;
  SemanticCircuit action;
};

/// Named instructions; names are unique.
template <typename Instruction>
struct InstructionSet {
  std::string name;

  /// Adds instruction; throws std::invalid_argument on a duplicate name.
  const Instruction &append(Instruction instruction) {
    if (find(instruction.get_name()))
      throw std::invalid_argument("instruction set already has '" +
                                  instruction.get_name() + "'");
    return instructions.emplace_back(std::move(instruction));
  }

  const std::vector<Instruction> &get_instructions() const {
    return instructions;
  }

  /// The index of the instruction with this name, if any.
  std::optional<std::size_t> index_of(std::string_view name) const {
    for (std::size_t i = 0; i < instructions.size(); ++i)
      if (instructions[i].get_name() == name)
        return i;
    return std::nullopt;
  }

  /// The instruction with this name, or null.
  const Instruction *find(std::string_view name) const {
    for (const Instruction &instruction : instructions)
      if (instruction.get_name() == name)
        return &instruction;
    return nullptr;
  }

private:
  std::vector<Instruction> instructions;
};

} // namespace v0
} // namespace cudaq::gadgets
