/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/circuit/standard.h"

#include <complex>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cudaq::gadgets {
inline namespace v0 {

using WireId = std::string;

// Pointers between objects are non-owning.
template <typename Instruction>
struct InstructionSet;
struct PhysicalInstructionDefinition;
using PhysicalInstructionSet = InstructionSet<PhysicalInstructionDefinition>;

/// Labels for nodes, by their index in a circuit.
struct DebugInfo {
  std::vector<std::pair<std::size_t, std::string>> labels;
};

/// Qubit and classical input and output wires, of a node or a circuit.
struct Wires {
  Wires(std::vector<WireId> q_in = {}, std::vector<WireId> q_out = {},
        std::vector<WireId> c_in = {}, std::vector<WireId> c_out = {})
      : q_in(std::move(q_in)), q_out(std::move(q_out)), c_in(std::move(c_in)),
        c_out(std::move(c_out)) {}

  std::vector<WireId> q_in;
  std::vector<WireId> q_out;
  std::vector<WireId> c_in;
  std::vector<WireId> c_out;

  Shape get_shape() const {
    return {q_in.size(), q_out.size(), c_in.size(), c_out.size()};
  }
};

/// An operation's wires. A node is identified by its index in its circuit.
struct Node {
  const Wires &get_wires() const { return wires; }

  /// Throws std::invalid_argument unless the wires have shape.
  void check_shape(const Shape &shape) const {
    if (wires.get_shape() != shape)
      throw std::invalid_argument("wire counts do not match the operation");
  }

protected:
  explicit Node(Wires wires) : wires(std::move(wires)) {}

private:
  Wires wires;
};

/// A semantic operation, with a payload if its opcode takes one.
struct OpcodeNode : Node {
  /// Throws std::invalid_argument if the payload or wires do not fit opcode.
  OpcodeNode(Opcode opcode, Wires wires,
             std::optional<Payload> payload = std::nullopt)
      : Node(std::move(wires)), opcode(opcode), payload(std::move(payload)) {
    check_shape(get_shape(opcode, this->payload));
  }
  Opcode get_opcode() const { return opcode; }
  const std::optional<Payload> &get_payload() const { return payload; }

private:
  Opcode opcode;
  std::optional<Payload> payload;
};

/// A call to the instruction at an index of an instruction set.
struct InstructionNode : Node {
  InstructionNode(std::size_t instruction, Wires wires)
      : Node(std::move(wires)), instruction(instruction) {}
  std::size_t get_instruction() const { return instruction; }

private:
  std::size_t instruction;
};

// TODO: Implement validate() for both circuits, throwing
// std::invalid_argument unless:
// - each node fits its operation's shape, as for append;
// - each wire is defined once, by an input or a node;
// - each qubit wire is consumed once, by a node or an output;
// - wires are read only after they are defined;
// - classical outputs are distinct;
// - unitary matrices are unitary.

/// A circuit of semantic operations.
struct SemanticCircuit {
  explicit SemanticCircuit(Wires wires = {}) : wires(std::move(wires)) {}

  const Wires &get_wires() const { return wires; }
  const std::vector<OpcodeNode> &get_nodes() const { return nodes; }

  /// Appends node and returns its index.
  std::size_t append(OpcodeNode node) {
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }

  /// Appends an OpcodeNode built from opcode, wires, and payload.
  std::size_t append(Opcode opcode, Wires wires,
                     std::optional<Payload> payload = std::nullopt) {
    return append(OpcodeNode(opcode, std::move(wires), std::move(payload)));
  }

  void validate() const {}

private:
  Wires wires;
  std::vector<OpcodeNode> nodes;
};

/// A circuit of calls to the instructions of an instruction set.
struct InstructionCircuit {
  explicit InstructionCircuit(const PhysicalInstructionSet &instruction_set,
                              Wires wires = {})
      : instruction_set(&instruction_set), wires(std::move(wires)) {}

  const PhysicalInstructionSet &get_instruction_set() const {
    return *instruction_set;
  }
  const Wires &get_wires() const { return wires; }
  const std::vector<InstructionNode> &get_nodes() const { return nodes; }

  /// Appends node and returns its index. Throws
  /// std::invalid_argument if its instruction is not in the set, or its wires
  /// do not fit the instruction.
  std::size_t append(InstructionNode node);

  /// Appends an InstructionNode built from instruction and wires.
  std::size_t append(std::size_t instruction, Wires wires) {
    return append(InstructionNode(instruction, std::move(wires)));
  }

  void validate() const {}

private:
  const PhysicalInstructionSet *instruction_set;
  Wires wires;
  std::vector<InstructionNode> nodes;
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
