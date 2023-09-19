/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/ADT/GraphCSR.h"

namespace cudaq {

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
