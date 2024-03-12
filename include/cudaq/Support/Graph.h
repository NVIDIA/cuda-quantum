/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/ADT/GraphCSR.h"

namespace cudaq {

/// Returns the shortest path from \p src to every other destination in
/// \p graph. The return vector `vec[i]` contains the next node in path to
/// `src`. If `vec[i] == src`, then it is either an immediate neighbor, or there
/// is no path to get there (i.e. the graph is bipartite).
mlir::SmallVector<GraphCSR::Node> getShortestPathsBFS(const GraphCSR &graph,
                                                      GraphCSR::Node src) {
  assert(src.isValid() && "Invalid source node");
  mlir::SmallVector<bool> discovered(graph.getNumNodes(), false);
  mlir::SmallVector<GraphCSR::Node> parents(graph.getNumNodes(), src);
  mlir::SmallVector<GraphCSR::Node> queue;
  queue.reserve(graph.getNumNodes());
  queue.push_back(src);
  std::size_t begin = 0;
  while (begin < queue.size()) {
    auto node = queue[begin++];
    for (auto neighbour : graph.getNeighbours(node)) {
      if (discovered[neighbour.index])
        continue;
      parents[neighbour.index] = node;
      discovered[neighbour.index] = true;
      queue.push_back(neighbour);
    }
  }
  return parents;
}

} // namespace cudaq
