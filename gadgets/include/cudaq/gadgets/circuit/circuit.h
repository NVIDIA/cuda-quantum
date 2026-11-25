/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/utils.h"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudaq::gadgets {
inline namespace v0 {

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

/// A circuit: boundary wires and nodes of NodeType, in order.
template <typename NodeType>
struct Circuit {
  static_assert(std::is_base_of_v<Node, NodeType>);

  virtual ~Circuit() = default;

  const Wires &get_wires() const { return wires; }
  const std::vector<NodeType> &get_nodes() const { return nodes; }

  /// Appends node and returns its index.
  virtual std::size_t append(NodeType node) {
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }

  /// TODO: Throw std::invalid_argument unless:
  /// - each node fits its operation's shape, as for append;
  /// - each wire is defined once, by an input or a node;
  /// - each qubit wire is consumed once, by a node or an output;
  /// - wires are read only after they are defined;
  /// - classical outputs are distinct;
  /// - unitary matrices are unitary.
  void validate() const {}

protected:
  explicit Circuit(Wires wires) : wires(std::move(wires)) {}

private:
  Wires wires;
  std::vector<NodeType> nodes;
};

/// Labels for nodes, by their index in a circuit.
struct DebugInfo {
  std::vector<std::pair<std::size_t, std::string>> labels;
};

} // namespace v0
} // namespace cudaq::gadgets
