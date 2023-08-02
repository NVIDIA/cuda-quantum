/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/Support/GraphPartioner.h"
#include "cudaq/Support/Plugin.h"

#include <map>
#include <metis.h>

using namespace cudaq;

namespace {

class MetisGraphPartitioner : public cudaq::GraphPartitioner {
private:
  /// @brief Map one of our Graph data types to a the METIS
  /// input format, (CSR). Must be converted from DAG to Undirected first
  std::tuple<std::vector<idx_t>, std::vector<idx_t>> toMetis(Graph graph) {

    // Convert to undirected version of the graph
    for (auto &[node, edges] : graph) {
      // auto edges = graph[node];
      for (auto &otherNode : edges) {
        auto &otherEdges = graph[otherNode];
        if (std::find(otherEdges.begin(), otherEdges.end(), node) ==
            otherEdges.end())
          otherEdges.push_back(node);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "\n\nUndirected View:\n");
    dumpGraph(graph);

    // Sort the nodes
    std::vector<GraphNode> nodes;
    sortNodes(graph, nodes);

    std::map<std::size_t, std::vector<std::size_t>> toUndirected;
    std::vector<idx_t> xAdj(1), adj;
    for (std::size_t numVertices = 0; auto &node : nodes) {
      auto edges = graph[node];
      auto numEdges = edges.size();
      for (std::size_t i = 0; i < numEdges; i++)
        adj.emplace_back(edges[i].uniqueId);
      xAdj.emplace_back(xAdj[numVertices++] + numEdges);
    }
    return std::make_tuple(xAdj, adj);
  }

  /// @brief Given the number of partitions and the coloring of the
  /// graph nodes, return a vector of graphs, where each one represents
  /// a partition of the Graph.
  std::vector<Graph> createPartitionGraphs(const Graph &graph, std::size_t k,
                                           std::vector<idx_t> &part) {

    std::map<std::size_t, Graph> partitioned;
    for (std::size_t i = 0; i < k; i++)
      partitioned.insert({i, Graph{}});

    for (auto &[node, edges] : graph) {
      auto p = part[node.uniqueId];
      auto &g = partitioned[p];

      g.insert({node, edges});
    }

    // Convert to a vector
    std::vector<Graph> graphs;
    for (auto &[id, g] : partitioned)
      graphs.emplace_back(g);

    LLVM_DEBUG(llvm::dbgs() << "Partitioned Graphs:\n");
    for (auto &[p, g] : partitioned)
      dumpGraph(g);

    return graphs;
  }

public:
  virtual ~MetisGraphPartitioner() = default;

  std::vector<cudaq::Graph> partition(const Graph &graph,
                                      std::size_t numPartitions) override {

    // Convert this graph rep to Metis input deck
    idx_t nNodes = graph.size(), nWeights = 1, objval, k = numPartitions;
    real_t im = 10.0; // FIXME This is a magic number...

    // Map our graph to the Metis input format (CSR)
    auto [xAdj, adj] = toMetis(graph);
    for (auto &x : xAdj)
      LLVM_DEBUG(llvm::dbgs() << x << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");
    for (auto &x : adj)
      LLVM_DEBUG(llvm::dbgs() << x << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n\n");

    // Weights of vertices
    // if all weights are equal then can be set to NULL
    std::vector<idx_t> vwgt(nNodes * nWeights, 1), part(nNodes, 0);

    // Partition the graph into k parts
    [[maybe_unused]] int ret = METIS_PartGraphKway(
        &nNodes, &nWeights, xAdj.data(), adj.data(), NULL, NULL, NULL, &k, NULL,
        &im, NULL, &objval, part.data());

    for (unsigned part_i = 0; part_i < part.size(); part_i++) {
      LLVM_DEBUG(llvm::dbgs() << "Node " << part_i << " is in partition "
                              << part[part_i] << "\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Create new Graphs for each partition
    return createPartitionGraphs(graph, k, part);
  }
};
} // namespace

// Load it via the runtime
CUDAQ_REGISTER_TYPE(cudaq::GraphPartitioner, MetisGraphPartitioner, metis)

// Make it possible to load with cudaq-opt
CUDAQ_REGISTER_MLIR_PLUGIN(metis, []() {
  if (!cudaq::registry::isRegistered<cudaq::GraphPartitioner>("metis"))
    cudaq::GraphPartitioner::RegistryType::Add<MetisGraphPartitioner> X("metis",
                                                                        "");
})