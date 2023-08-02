/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "cut-quake"

#include "common/Registry.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace cudaq {

/// @brief A GraphNode represents a single vertex
/// in a directed acyclic graph. It keeps track of the
/// MLIR Operation it represents, a unique integer ID, and
/// an optional string name.
struct GraphNode {
  /// @brief The MLIR Operation this GraphNode represents.
  Operation *op;

  /// @brief Unique ID for this node
  std::size_t uniqueId = 0;

  /// @brief The name of this Graph Node.
  std::string name = "";

  /// @brief Move constructor
  GraphNode(GraphNode &&) = default;

  /// @brief Copy Constructor
  GraphNode(const GraphNode &other)
      : op(other.op), uniqueId(other.uniqueId), name(other.name) {}

  /// @brief Constructor, create from MLIR Operation and unique ID
  GraphNode(Operation *o, std::size_t i) : op(o), uniqueId(i) {}

  /// @brief Constructor, create from name and unique ID
  GraphNode(std::string n, std::size_t i) : op(nullptr), uniqueId(i), name(n) {}

  /// @brief Assignment operator
  GraphNode &operator=(const GraphNode &other) {
    op = other.op;
    uniqueId = other.uniqueId;
    name = other.name;
    return *this;
  }

  /// @brief Return true if this GraphNode is equal to the input GraphNode.
  bool operator==(const GraphNode &node) const {
    return op == node.op && uniqueId == node.uniqueId;
  }

  /// @brief Return the name of this GraphNode
  std::string getName() const {
    if (op == nullptr)
      return name + ":" + std::to_string(uniqueId);

    return op->getName().getStringRef().str() + ":" + std::to_string(uniqueId);
  }
};

/// @brief Hash functor for GraphNodes in a unordered_map
struct GraphNodeHash {
  std::size_t operator()(const GraphNode &node) const {
    std::size_t seed = reinterpret_cast<intptr_t>(node.op);
    seed ^= std::hash<std::size_t>()(node.uniqueId) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

/// @brief We represent directed acylic graphs as a
/// collection of GraphNodes, with each GraphNode mapping
/// to a vector of GraphNodes that it connects to (representative of
/// its edges)
using Graph =
    std::unordered_map<GraphNode, std::vector<GraphNode>, GraphNodeHash>;

/// @brief Dump the given graph to stderr.
inline void dumpGraph(Graph &graph) {
  for (auto &[node, edges] : graph) {
    std::string name = node.getName();
    LLVM_DEBUG(llvm::dbgs() << name << " --> [");
    for (std::size_t i = 0; auto &e : edges) {
      LLVM_DEBUG(llvm::dbgs()
                 << e.getName() << (i++ == edges.size() - 1 ? "" : ", "));
    }
    LLVM_DEBUG(llvm::dbgs() << "]\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

/// @brief Sort the nodes in the Graph by unique ID. Write the
/// sorted GraphNodes to the input vector reference.
inline void sortNodes(Graph &graph, std::vector<GraphNode> &sortedNodes) {
  for (auto &[node, edges] : graph)
    sortedNodes.push_back(node);
  std::sort(sortedNodes.begin(), sortedNodes.end(),
            [](const GraphNode &n, const GraphNode &m) {
              return n.uniqueId < m.uniqueId;
            });
}

/// @brief The GraphPartitioner presents an interface for concrete
/// library implementations that take an input graph and number of partitions
/// and partitions the graph into subgraphs that are returned as a vector.
class GraphPartitioner : public registry::RegisteredType<GraphPartitioner> {
public:
  /// @brief Partition the input graph into `numPartitions`.
  virtual std::vector<Graph> partition(const Graph &graph,
                                       std::size_t numPartitions) = 0;
  virtual ~GraphPartitioner() = default;
};
} // namespace cudaq