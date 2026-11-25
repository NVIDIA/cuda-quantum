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

using Id = std::string;
using WireId = std::string;

// Pointers between objects are non-owning.
template <typename Instruction>
struct InstructionSet;
struct PhysicalInstructionDefinition;
using PhysicalInstructionSet = InstructionSet<PhysicalInstructionDefinition>;

struct DebugInfo {
  std::vector<std::pair<Id, std::string>> labels;
};

/// An operation: an opcode and its wires, plus a payload for a semantic
/// instruction that takes one. In a circuit over a physical instruction set,
/// the opcode indexes the set; otherwise it is an Opcode.
struct Node {
  Node(Id id, std::size_t opcode, std::vector<WireId> q_in,
       std::vector<WireId> q_out, std::vector<WireId> c_in = {},
       std::vector<WireId> c_out = {},
       std::optional<Payload> payload = std::nullopt)
      : id(std::move(id)), opcode(opcode), q_in(std::move(q_in)),
        q_out(std::move(q_out)), c_in(std::move(c_in)), c_out(std::move(c_out)),
        payload(std::move(payload)) {}
  Node(Id id, Opcode opcode, std::vector<WireId> q_in,
       std::vector<WireId> q_out, std::vector<WireId> c_in = {},
       std::vector<WireId> c_out = {},
       std::optional<Payload> payload = std::nullopt)
      : Node(std::move(id), static_cast<std::size_t>(opcode), std::move(q_in),
             std::move(q_out), std::move(c_in), std::move(c_out),
             std::move(payload)) {}

  const Id &get_id() const { return id; }
  std::size_t get_opcode() const { return opcode; }
  const std::vector<WireId> &get_q_in() const { return q_in; }
  const std::vector<WireId> &get_q_out() const { return q_out; }
  const std::vector<WireId> &get_c_in() const { return c_in; }
  const std::vector<WireId> &get_c_out() const { return c_out; }
  const std::optional<Payload> &get_payload() const { return payload; }
  /// The numbers of wires.
  Shape get_shape() const {
    return {q_in.size(), q_out.size(), c_in.size(), c_out.size()};
  }

private:
  Id id;
  std::size_t opcode;
  std::vector<WireId> q_in;
  std::vector<WireId> q_out;
  std::vector<WireId> c_in;
  std::vector<WireId> c_out;
  std::optional<Payload> payload;
};

/// A circuit over a physical instruction set, or over the semantic
/// instructions (Opcode) when instruction_set is null.
struct Circuit {
  explicit Circuit(std::vector<WireId> q_in = {},
                   std::vector<WireId> q_out = {},
                   std::vector<WireId> c_in = {},
                   std::vector<WireId> c_out = {},
                   const PhysicalInstructionSet *instruction_set = nullptr)
      : instruction_set(instruction_set), q_in(std::move(q_in)),
        q_out(std::move(q_out)), c_in(std::move(c_in)),
        c_out(std::move(c_out)) {}

  const PhysicalInstructionSet *get_instruction_set() const {
    return instruction_set;
  }
  const std::vector<WireId> &get_q_in() const { return q_in; }
  const std::vector<WireId> &get_q_out() const { return q_out; }
  const std::vector<WireId> &get_c_in() const { return c_in; }
  const std::vector<WireId> &get_c_out() const { return c_out; }
  const std::vector<Node> &get_nodes() const { return nodes; }

  /// The shape of node's operation. Throws std::invalid_argument if the
  /// opcode is not in the instruction set, or the payload does not fit it;
  /// only semantic instructions take a payload.
  Shape get_shape(const Node &node) const;

  /// Appends node after checking its wires against its operation's shape, and
  /// returns its ID and outcomes.
  /// TODO: Check node as validate() would.
  std::pair<Id, std::vector<OutcomeRef>> append(Node node);

  /// TODO: Throw std::invalid_argument unless:
  /// - each node fits its operation's shape, as for append;
  /// - each wire is defined once, by an input or a node;
  /// - each qubit wire is consumed once, by a node or an output;
  /// - wires are read only after they are defined;
  /// - classical outputs are distinct;
  /// - unitary matrices are unitary.
  void validate() const {}

private:
  const PhysicalInstructionSet *instruction_set;
  std::vector<WireId> q_in;
  std::vector<WireId> q_out;
  std::vector<WireId> c_in;
  std::vector<WireId> c_out;
  std::vector<Node> nodes;
};

/// A physical instruction, defined by an action of standard operations whose
/// boundary wires bind to its ports by index.
struct PhysicalInstructionDefinition {
  /// Validates action, which must be over the semantic instructions, takes
  /// its shape from the boundary, and derives port_map by tracing each input
  /// to the output it reaches.
  /// TODO: Queries for port flags (prepared, reset, discarded, inactive).
  PhysicalInstructionDefinition(std::string name, Circuit action);

  const std::string &get_name() const { return name; }
  const Shape &get_shape() const { return shape; }
  /// (input port, output port) for each qubit passing through.
  const std::vector<std::pair<std::size_t, std::size_t>> &get_port_map() const {
    return port_map;
  }
  const Circuit &get_action() const { return action; }

private:
  std::string name;
  Shape shape;
  std::vector<std::pair<std::size_t, std::size_t>> port_map;
  Circuit action;
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
