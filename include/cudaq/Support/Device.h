/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/ADT/GraphCSR.h"
#include "cudaq/Support/Graph.h"
#include "llvm/Support/MemoryBuffer.h"

namespace cudaq {

/// The `Device` class represents a device topology with qubits and connections
/// between qubits. It contains various methods to construct the device based on
/// canned geometries, and it contains helper methods to determine paths between
/// qubits.
class Device {
public:
  using Qubit = GraphCSR::Node;
  using Path = mlir::SmallVector<Qubit>;

  /// Read device connectivity info from a file. The input format is the same
  /// as the Graph dump() format.
  static Device file(llvm::StringRef filename) {
    Device device;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileBuffer =
        llvm::MemoryBuffer::getFile(filename);
    if (std::error_code EC = fileBuffer.getError()) {
      llvm::errs() << "Error reading file: " << EC.message() << "\n";
      return device;
    }

    llvm::StringRef fileContent = fileBuffer->get()->getBuffer();
    while (!fileContent.empty()) {
      auto [line, rest] = fileContent.split('\n');
      fileContent = rest;

      if (line.consume_front("Number of nodes:")) {
        line = line.ltrim();
        unsigned numQubits = 0;
        if (!line.consumeInteger(/*Radix=*/10, numQubits)) {
          for (unsigned i = 0u; i < numQubits; ++i)
            device.topology.createNode();
        }
      } else {
        // Parse edges
        unsigned v1 = 0;
        line = line.ltrim();
        if (!line.consumeInteger(/*Radix=*/10, v1)) {
          line = line.ltrim();
          if (line.consume_front("--> {")) {
            line = line.ltrim();
            unsigned v2 = 0;
            while (!line.consumeInteger(10, v2)) {
              // Create an edge, but make sure it doesn't already exist
              bool edgeAlreadyExists = false;
              for (auto edge : device.topology.getNeighbours(Qubit(v1))) {
                if (edge == Qubit(v2)) {
                  edgeAlreadyExists = true;
                  break;
                }
              }
              if (!edgeAlreadyExists)
                device.topology.addEdge(Qubit(v1), Qubit(v2));
              // Prepare for next iteration (removing comma)
              line = line.ltrim(" \t\n\v\f\r,");
            }
          }
        }
      }
    }

    device.computeAllPairShortestPaths();
    return device;
  }

  /// Create a device with a path topology.
  ///
  ///  0 -- 1 -- ... -- N
  ///
  static Device path(unsigned numQubits) {
    assert(numQubits > 0);
    Device device;
    device.topology.createNode();
    for (unsigned i = 1u; i < numQubits; ++i) {
      device.topology.createNode();
      device.topology.addEdge(Qubit(i - 1), Qubit(i));
    }
    device.computeAllPairShortestPaths();
    return device;
  }

  // Create a device with a ring topology.
  ///
  ///  0 -- 1 -- ... -- N
  ///  |________________|
  ///
  static Device ring(unsigned numQubits) {
    assert(numQubits > 0);
    Device device;
    device.topology.createNode();
    for (unsigned i = 0u; i < numQubits; ++i) {
      device.topology.createNode();
      device.topology.addEdge(Qubit(i), Qubit((i + 1) % numQubits));
    }
    device.computeAllPairShortestPaths();
    return device;
  }

  /// Create a device with star topology.
  ///
  ///    2  3  4
  ///     \ | /
  ///  1 -- 0 -- 5
  ///     / | \
  ///    N  7  6
  ///
  /// @param numQubits Number of qubits in topology
  /// @param centerQubit 0-based ID of center qubit (default 0)
  static Device star(unsigned numQubits, unsigned centerQubit = 0) {
    Device device;

    // Create nodes
    for (unsigned i = 0u; i < numQubits; ++i)
      device.topology.createNode();

    // Create edges
    for (unsigned i = 0u; i < numQubits; ++i)
      if (i != centerQubit)
        device.topology.addEdge(Qubit(centerQubit), Qubit(i));

    device.computeAllPairShortestPaths();
    return device;
  }

  /// Create a device with a grid topology.
  ///
  ///  0 -- 1 -- 2
  ///  |    |    |
  ///  3 -- 4 -- 5
  ///  |    |    |
  ///  6 -- 7 -- 8
  ///
  static Device grid(unsigned width, unsigned height) {
    Device device;
    for (unsigned i = 0u, end = width * height; i < end; ++i)
      device.topology.createNode();
    for (unsigned x = 0u; x < width; ++x) {
      for (unsigned y = 0u; y < height; ++y) {
        unsigned base = x + (y * width);
        Qubit q0(base);
        if (x < width - 1)
          device.topology.addEdge(q0, Qubit(base + 1));
        if (y < height - 1)
          device.topology.addEdge(q0, Qubit(base + width));
      }
    }
    device.computeAllPairShortestPaths();
    return device;
  }

  /// TODO: Implement a method to load device info from a file.

  /// Returns the number of physical qubits in the device.
  unsigned getNumQubits() const { return topology.getNumNodes(); }

  /// Returns the distance between two qubits.
  unsigned getDistance(Qubit src, Qubit dst) const {
    unsigned pairID = getPairID(src.index, dst.index);
    return src == dst ? 0 : shortestPaths[pairID].size() - 1;
  }

  mlir::ArrayRef<Qubit> getNeighbours(Qubit src) const {
    return topology.getNeighbours(src);
  }

  bool areConnected(Qubit q0, Qubit q1) const {
    return getDistance(q0, q1) == 1;
  }

  /// Returns a shortest path between two qubits.
  Path getShortestPath(Qubit src, Qubit dst) const {
    unsigned pairID = getPairID(src.index, dst.index);
    if (src.index > dst.index)
      return Path(llvm::reverse(shortestPaths[pairID]));
    return Path(shortestPaths[pairID]);
  }

  void dump(llvm::raw_ostream &os = llvm::errs()) const {
    os << "Graph:\n";
    topology.dump(os);
    os << "\nShortest Paths:\n";
    for (unsigned src = 0; src < getNumQubits(); ++src)
      for (unsigned dst = 0; dst < getNumQubits(); ++dst) {
        auto path = getShortestPath(Qubit(src), Qubit(dst));
        os << '(' << src << ", " << dst << ") : {";
        llvm::interleaveComma(path, os);
        os << "}\n";
      }
  }

private:
  using PathRef = mlir::ArrayRef<Qubit>;

  /// Returns a unique id for a pair of values (`u` and `v`). `getPairID(u, v)`
  /// will be equal to `getPairID(v, u)`.
  unsigned getPairID(unsigned u, unsigned v) const {
    if (u > v)
      std::swap(u, v);
    return (u * getNumQubits()) - (((u - 1) * u) / 2) + v - u;
  }

  /// Compute the shortest path between every qubit. This assumes that there
  /// exists at least one path between every source and destination pair. I.e.
  /// the graph cannot be bipartite.
  void computeAllPairShortestPaths() {
    std::size_t numNodes = topology.getNumNodes();
    shortestPaths.resize(numNodes * (numNodes + 1) / 2);
    mlir::SmallVector<Qubit> path(numNodes);
    for (unsigned n = 0; n < numNodes; ++n) {
      auto parents = getShortestPathsBFS(topology, Qubit(n));
      // Reconstruct the paths
      for (auto m = n + 1; m < numNodes; ++m) {
        path.clear();
        path.push_back(Qubit(m));
        auto p = parents[m];
        while (p != Qubit(n)) {
          path.push_back(p);
          p = parents[p.index];
        }
        path.push_back(Qubit(n));
        pathsData.append(path.rbegin(), path.rend());
        shortestPaths[getPairID(n, m)] =
            PathRef(pathsData.end() - path.size(), pathsData.end());
      }
    }
  }

  /// Device nodes (qubits) and edges (connections)
  GraphCSR topology;

  /// List of shortest path from/to every source/destination
  mlir::SmallVector<PathRef> shortestPaths;

  /// Storage for `PathRef`'s in `shortestPaths`
  mlir::SmallVector<Qubit> pathsData;
};

} // namespace cudaq
