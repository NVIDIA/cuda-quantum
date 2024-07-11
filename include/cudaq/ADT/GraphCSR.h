/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/Handle.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

namespace cudaq {

/// Compressed Sparse Row Format (CSR) for Representing Graphs
///
/// This way of representing graphs is based on a technique that originated
/// in HPC as a way to represent sparse matrices. The format is more compact and
/// is laid out more contiguously in memory than other forms, e.g., as adjacency
/// lists, which eliminates most space overheads and reduces random memory
/// accesses.
///
/// The price payed for these advantages is reduced flexibility and complexity
/// (cognitive overhead):
///   * Adding new edges is inefficient (see `addEdgeImpl`method)
///   * The implementation is trickier than other forms.
///
/// Since adding new edges is inefficient, this class suitable for graphs whose
/// structure is fixed and given all at once.
class GraphCSR {
  using Offset = unsigned;

public:
  struct Node : Handle {
    using Handle::Handle;
  };

  GraphCSR() = default;

  /// Creates a new node in the graph and returns its unique identifier.
  Node createNode() {
    Node node(getNumNodes());
    nodeOffsets.push_back(edges.size());
    return node;
  }

  void addEdge(Node src, Node dst, bool undirected = true) {
    assert(src.isValid() && "Invalid source node");
    assert(dst.isValid() && "Invalid destination node");
    addEdgeImpl(src, dst);
    if (undirected)
      addEdgeImpl(dst, src);
  }

  std::size_t getNumNodes() const { return nodeOffsets.size(); }

  std::size_t getNumEdges() const { return edges.size(); }

  mlir::ArrayRef<Node> getNeighbours(Node node) const {
    assert(node.isValid() && "Invalid node");
    auto begin = edges.begin() + nodeOffsets[node.index];
    auto end = node == Node(getNumNodes() - 1)
                   ? edges.end()
                   : edges.begin() + nodeOffsets[node.index + 1];
    return mlir::ArrayRef<Node>(begin, end);
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &os = llvm::errs()) const {
    if (getNumNodes() == 0) {
      os << "Empty graph.\n";
      return;
    }

    os << "Number of nodes: " << getNumNodes() << '\n';
    os << "Number of edges: " << getNumEdges() << '\n';
    std::size_t lastID = getNumNodes() - 1;
    for (std::size_t id = 0; id < lastID; ++id) {
      os << id << " --> {";
      for (Offset j = nodeOffsets[id], end = nodeOffsets[id + 1]; j < end; ++j)
        os << edges[j] << (j == end - 1 ? "" : ", ");
      os << "}\n";
    }

    // Handle last node
    os << lastID << " --> {";
    for (Offset j = nodeOffsets[lastID], end = edges.size(); j < end; ++j)
      os << edges[j] << (j == end - 1 ? "" : ", ");
    os << "}\n";
  }

  /// Dumps to `.gv` (aka `.dot`) format for rendering in external Graphviz
  /// viewer, e.g. https://dreampuf.github.io/GraphvizOnline/
  LLVM_DUMP_METHOD void
  dumpToGV(const std::string &graphName = "cudaqMappingGraph",
           llvm::raw_ostream &os = llvm::errs()) const {
    os << "digraph " << graphName << " {\n";
    std::size_t srcNode = 0;
    for (std::size_t edge = 0; edge < getNumEdges(); edge++) {
      // Check to see if we have moved to the next source node
      while (srcNode + 1 < getNumNodes() && nodeOffsets[srcNode + 1] == edge)
        srcNode++;
      os << "  q" << srcNode << " -> q" << edges[edge] << "\n";
    }
    os << "}\n";
  }

private:
  void addEdgeImpl(Node src, Node dst) {
    // If the source node is the last node, we just need push-back edges.
    if (src == Node(getNumNodes() - 1)) {
      edges.push_back(dst);
      return;
    }

    // Insert the destination node in the offset.
    edges.insert(edges.begin() + nodeOffsets[src.index], dst);

    // Update the offsets of all nodes that have an ID greater than `src`.
    src.index += 1;
    std::transform(nodeOffsets.begin() + src.index, nodeOffsets.end(),
                   nodeOffsets.begin() + src.index,
                   [](Offset offset) { return offset + 1; });
  }

  /// Each entry in this vector contains the starting index in the edge array
  /// where the edges from that node are stored.
  mlir::SmallVector<Offset> nodeOffsets;

  // Stores the destination vertices of each edge.
  mlir::SmallVector<Node> edges;
};

} // namespace cudaq
