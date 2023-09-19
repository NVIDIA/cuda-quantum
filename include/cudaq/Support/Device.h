/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/ADT/GraphCSR.h"
#include "cudaq/Support/Graph.h"

namespace cudaq {

class Device {
public:
  using Qubit = GraphCSR::Node;
  using Path = mlir::SmallVector<Qubit>;

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
  static Device star(unsigned numQubits) {
    Device device;
    Qubit center = device.topology.createNode();
    for (unsigned i = 1u; i < numQubits; ++i) {
      device.topology.createNode();
      device.topology.addEdge(center, Qubit(i));
    }
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
    return getDistance(q0, q1) == 1 ? true : false;
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

  unsigned getPairID(unsigned u, unsigned v) const {
    if (u > v)
      std::swap(u, v);
    return (u * getNumQubits()) - (((u - 1) * u) / 2) + v - u;
  }

  void computeAllPairShortestPaths() {
    std::size_t numNodes = topology.getNumNodes();
    shortestPaths.resize(numNodes * (numNodes + 1) / 2);
    mlir::SmallVector<Qubit> path;
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
        std::copy(path.rbegin(), path.rend(), std::back_inserter(pathsData));
        shortestPaths[getPairID(n, m)] =
            PathRef(pathsData.end() - path.size(), pathsData.end());
      }
    }
  }

  GraphCSR topology;
  mlir::SmallVector<PathRef> shortestPaths;
  mlir::SmallVector<Qubit> pathsData;
};

} // namespace cudaq
