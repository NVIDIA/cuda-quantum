/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/circuit/pauli.h"
#include "cudaq/gadgets/circuit/standard.h"

#include <complex>
#include <cstddef>
#include <memory>
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
using Param = std::string;
using OutcomeRef = std::string;
/// A constant or a parameter name.
using ParamValue = std::variant<double, Param>;

// Pointers between objects are non-owning.
template <typename Instruction>
struct InstructionSet;
template <typename Instruction>
struct BasicCircuit;
struct PhysicalInstructionDefinition;
using PhysicalInstructionSet = InstructionSet<PhysicalInstructionDefinition>;
using Circuit = BasicCircuit<PhysicalInstructionDefinition>;

struct DebugInfo {
  std::vector<std::pair<Id, std::string>> labels;
};

/// XOR of distinct outcomes and a constant offset.
struct ParityExpr {
  std::vector<OutcomeRef> refs;
  bool offset = false;
};

using Predicate = ParityExpr;

enum class NodeKind : u32 {
  Instruction,
  GadgetCall,
  ConditionalPauli,
  // Atomic nodes; keep contiguous.
  Opcode,
  Idle,
  RandomBit,
  PauliUnitary,
  Unitary,
  JointMeasurement
};

/// An operation with an ID; wires bind to ports by position. Supports
/// LLVM-style isa/dyn_cast through classof.
struct Node {
  virtual ~Node() = default;
  NodeKind get_kind() const { return kind; }
  const Id &get_id() const { return id; }
  const std::vector<WireId> &get_q_in() const { return q_in; }
  const std::vector<WireId> &get_q_out() const { return q_out; }
  const std::vector<WireId> &get_c_in() const { return c_in; }
  const std::vector<WireId> &get_c_out() const { return c_out; }

  /// The qubit output continuing qubit input k, if any; output k by default.
  virtual std::optional<std::size_t> trace(std::size_t k,
                                           const Circuit &) const {
    return k < q_out.size() ? std::optional<std::size_t>(k) : std::nullopt;
  }

protected:
  Node(NodeKind kind, Id id, std::vector<WireId> q_in,
       std::vector<WireId> q_out, std::vector<WireId> c_in,
       std::vector<WireId> c_out)
      : kind(kind), id(std::move(id)), q_in(std::move(q_in)),
        q_out(std::move(q_out)), c_in(std::move(c_in)),
        c_out(std::move(c_out)) {}

private:
  NodeKind kind;
  Id id;
  std::vector<WireId> q_in;
  std::vector<WireId> q_out;
  std::vector<WireId> c_in;
  std::vector<WireId> c_out;
};

/// Call to a named instruction or gadget.
template <NodeKind Kind>
struct OccurrenceNode : Node {
  /// Each declared param of the callee, bound to a value.
  using Params = std::vector<std::pair<Param, ParamValue>>;

  OccurrenceNode(Id id, std::string callee, std::vector<WireId> q_in,
                 std::vector<WireId> q_out, std::vector<WireId> c_in = {},
                 std::vector<WireId> c_out = {}, Params params = {})
      : Node(Kind, std::move(id), std::move(q_in), std::move(q_out),
             std::move(c_in), std::move(c_out)),
        callee(std::move(callee)), params(std::move(params)) {}
  /// The callee's name.
  const std::string &get_callee() const { return callee; }
  const Params &get_params() const { return params; }
  static bool classof(const Node *node) { return node->get_kind() == Kind; }

private:
  std::string callee;
  Params params;
};

/// Call to a physical instruction.
struct InstructionNode : OccurrenceNode<NodeKind::Instruction> {
  using OccurrenceNode::OccurrenceNode;

  /// Follows the callee's port map.
  std::optional<std::size_t> trace(std::size_t k,
                                   const Circuit &circuit) const override;
};

/// Applies pauli when predicate is one.
struct ConditionalPauliNode : Node {
  ConditionalPauliNode(Id id, PauliString pauli, Predicate predicate,
                       std::vector<WireId> q_in, std::vector<WireId> q_out)
      : Node(NodeKind::ConditionalPauli, std::move(id), std::move(q_in),
             std::move(q_out), {}, {}),
        pauli(std::move(pauli)), predicate(std::move(predicate)) {
    if (get_q_in().size() != this->pauli.ops.size() ||
        get_q_out().size() != this->pauli.ops.size())
      throw std::invalid_argument("wire counts do not match the Pauli");
  }
  const PauliString &get_pauli() const { return pauli; }
  const Predicate &get_predicate() const { return predicate; }
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::ConditionalPauli;
  }

private:
  PauliString pauli;
  Predicate predicate;
};

/// An elementary operation, not a call.
struct AtomicNode : Node {
  static bool classof(const Node *node) {
    return node->get_kind() >= NodeKind::Opcode &&
           node->get_kind() <= NodeKind::JointMeasurement;
  }

protected:
  AtomicNode(NodeKind kind, Id id, std::vector<WireId> q_in,
             std::vector<WireId> q_out, std::vector<WireId> c_out,
             std::size_t num_q_in, std::size_t num_q_out, std::size_t num_c_out)
      : Node(kind, std::move(id), std::move(q_in), std::move(q_out), {},
             std::move(c_out)) {
    if (get_q_in().size() != num_q_in || get_q_out().size() != num_q_out ||
        get_c_out().size() != num_c_out)
      throw std::invalid_argument("wire counts do not match the operation");
  }
};

/// A standard operation; its contract gives the wire counts and param.
struct OpcodeNode : AtomicNode {
  OpcodeNode(Id id, Opcode opcode, std::vector<WireId> q_in,
             std::vector<WireId> q_out, std::vector<WireId> c_out = {},
             std::optional<ParamValue> param = std::nullopt)
      : AtomicNode(NodeKind::Opcode, std::move(id), std::move(q_in),
                   std::move(q_out), std::move(c_out), contract(opcode).q_in,
                   contract(opcode).q_out, contract(opcode).c_out),
        opcode(opcode), param(std::move(param)) {
    if (this->param.has_value() != (contract(opcode).param != nullptr))
      throw std::invalid_argument(this->param ? "opcode does not take a param"
                                              : "opcode needs a param");
  }
  Opcode get_opcode() const { return opcode; }
  const std::optional<ParamValue> &get_param() const { return param; }
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::Opcode;
  }

private:
  Opcode opcode;
  std::optional<ParamValue> param;
};

/// Identity on its qubits.
struct IdleNode : AtomicNode {
  IdleNode(Id id, std::vector<WireId> q_in, std::vector<WireId> q_out)
      : AtomicNode(NodeKind::Idle, std::move(id), q_in, std::move(q_out), {},
                   q_in.size(), q_in.size(), 0) {}
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::Idle;
  }
};

/// Produces a uniformly random bit.
struct RandomBitNode : AtomicNode {
  RandomBitNode(Id id, WireId bit)
      : AtomicNode(NodeKind::RandomBit, std::move(id), {}, {}, {std::move(bit)},
                   0, 0, 1) {}
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::RandomBit;
  }
};

/// Applies pauli.
struct PauliUnitaryNode : AtomicNode {
  PauliUnitaryNode(Id id, PauliString pauli, std::vector<WireId> q_in,
                   std::vector<WireId> q_out)
      : AtomicNode(NodeKind::PauliUnitary, std::move(id), std::move(q_in),
                   std::move(q_out), {}, pauli.ops.size(), pauli.ops.size(), 0),
        pauli(std::move(pauli)) {}
  const PauliString &get_pauli() const { return pauli; }
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::PauliUnitary;
  }

private:
  PauliString pauli;
};

/// Applies a row-major 2^n x 2^n matrix to n qubits.
/// TODO: Variable width needs contract support, or fixed n (e.g. 1-3).
struct UnitaryNode : AtomicNode {
  UnitaryNode(Id id, std::vector<std::complex<double>> matrix,
              std::vector<WireId> q_in, std::vector<WireId> q_out)
      : AtomicNode(NodeKind::Unitary, std::move(id), q_in, std::move(q_out), {},
                   q_in.size(), q_in.size(), 0),
        matrix(std::move(matrix)) {
    if (this->matrix.size() != std::size_t{1} << (2 * get_q_in().size()))
      throw std::invalid_argument("matrix size does not match the qubits");
  }
  const std::vector<std::complex<double>> &get_matrix() const { return matrix; }
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::Unitary;
  }

private:
  std::vector<std::complex<double>> matrix;
};

/// Non-destructive measurement of a Hermitian pauli.
struct JointMeasurementNode : AtomicNode {
  JointMeasurementNode(Id id, PauliString pauli, std::vector<WireId> q_in,
                       std::vector<WireId> q_out, WireId outcome)
      : AtomicNode(NodeKind::JointMeasurement, std::move(id), std::move(q_in),
                   std::move(q_out), {std::move(outcome)}, pauli.ops.size(),
                   pauli.ops.size(), 1),
        pauli(std::move(pauli)) {
    if (this->pauli.phase_exponent % 2)
      throw std::invalid_argument("measured Pauli must be Hermitian");
  }
  const PauliString &get_pauli() const { return pauli; }
  static bool classof(const Node *node) {
    return node->get_kind() == NodeKind::JointMeasurement;
  }

private:
  PauliString pauli;
};

/// A circuit calling Instruction entries of an instruction set. In a gadget
/// circuit, qubit wires are code blocks.
template <typename Instruction>
struct BasicCircuit {
  std::string name;
  /// Null for a circuit of standard operations.
  const InstructionSet<Instruction> *instruction_set = nullptr;
  std::vector<WireId> q_in;
  std::vector<WireId> q_out;
  std::vector<WireId> c_in;
  std::vector<WireId> c_out;
  std::vector<std::unique_ptr<Node>> nodes;

  /// Appends node and returns its ID and outcomes.
  /// TODO: Check node as validate() would.
  std::pair<Id, std::vector<OutcomeRef>> append(std::unique_ptr<Node> node) {
    auto result = std::make_pair(node->get_id(), node->get_c_out());
    nodes.push_back(std::move(node));
    return result;
  }

  /// TODO: Throw std::invalid_argument unless:
  /// - each wire is defined once, by an input or a node;
  /// - each qubit wire is consumed once, by a node or an output;
  /// - wires and predicate bits are read only after they are defined;
  /// - with an instruction set, nodes are calls matching its signatures;
  ///   without one, they are standard operations;
  /// - parameter bindings match declared params;
  /// - predicate refs and classical outputs are distinct;
  /// - unitary matrices are unitary;
  /// - in a gadget circuit, code blocks match the gadgets' codes.
  void validate() const {}
};

/// A physical instruction, defined by an action of standard operations whose
/// boundary wires bind to its ports by index.
struct PhysicalInstructionDefinition {
  /// Validates action and derives the signature by tracing each input:
  /// - port_map pairs it with the output it reaches;
  /// - an input that reaches a discard is discarded;
  /// - an output no input reaches is prepared;
  /// - an output reached through a reset is reset;
  /// - an input wire that is also an output wire is inactive on both ports.
  PhysicalInstructionDefinition(std::string name, Circuit action);

  const std::string &get_name() const { return name; }
  /// Signature: flags per port.
  const std::vector<PortFlags> &get_q_in() const { return q_in; }
  const std::vector<PortFlags> &get_q_out() const { return q_out; }
  u32 get_c_in() const { return c_in; }
  u32 get_c_out() const { return c_out; }
  const std::vector<Param> &get_params() const { return params; }
  /// (input port, output port) for each qubit passing through.
  const std::vector<std::pair<u32, u32>> &get_port_map() const {
    return port_map;
  }
  const Circuit &get_action() const { return action; }

private:
  std::string name;
  std::vector<PortFlags> q_in;
  std::vector<PortFlags> q_out;
  u32 c_in = 0;
  u32 c_out = 0;
  std::vector<Param> params;
  std::vector<std::pair<u32, u32>> port_map;
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

/// One instruction per opcode, named as in its contract.
const PhysicalInstructionSet &standard_instruction_set();

} // namespace v0
} // namespace cudaq::gadgets
