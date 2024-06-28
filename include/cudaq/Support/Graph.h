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
inline mlir::SmallVector<GraphCSR::Node> getShortestPathsBFS(const GraphCSR &graph,
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


// Function that implements Dijkstra's single source
// shortest path algorithm for a graph

inline mlir::SmallVector<GraphCSR::Node> dijkstra(const GraphCSR &graph,
                                                      GraphCSR::Node src)
{
  mlir::SmallVector<bool> discovered(graph.getNumNodes(), false);
  mlir::SmallVector<GraphCSR::Node> parents(graph.getNumNodes(), src);
  mlir::SmallVector<int> distance(graph.getNumNodes(), INT_MAX);
  mlir::SmallVector<GraphCSR::Node> queue;
  queue.reserve(graph.getNumNodes());
  queue.push_back(src);
  distance[src.index]=0;
  //std::size_t begin = 0;
  for (std::size_t i=0; i<graph.getNumNodes();i++){
    int min=INT_MAX,min_index=0;
    for (std::size_t v=0;v<graph.getNumNodes();v++){
      if (discovered[v]){
        continue;
      }
      if (distance[v]<min){
        min=distance[v],min_index=v;
      }
    }
    discovered[min_index]=true;
    auto node=graph.retrieveNode(min_index);
    int count=0;
    mlir::ArrayRef<int> neighweights=graph.getNeighboursWeights(node);
    for (auto neighbour : graph.getNeighbours(node)){
      int new_dist= min+neighweights[count];
      if (discovered[neighbour.index]){
        continue;
      }
      if (new_dist< distance[neighbour.index]){
        distance[neighbour.index]=new_dist;
        parents[neighbour.index]=node;
      }
    }

  }

  return parents;
}


} // namespace cudaq
