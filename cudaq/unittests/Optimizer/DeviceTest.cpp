/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/Device.h"
#include "gtest/gtest.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace cudaq;

namespace {

using Qubit = Device::Qubit;

/// Round-trip a topology description through a temporary file. `Device::file`
/// is the only public constructor for an arbitrary (in particular,
/// disconnected) coupling graph, so the disconnected cases go through it.
Device deviceFromTopology(llvm::StringRef topology) {
  llvm::SmallString<128> path;
  int fd = -1;
  EXPECT_FALSE(llvm::sys::fs::createTemporaryFile("topology", "txt", fd, path));
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    os << topology;
  }
  Device device;
  if (llvm::Error error = Device::tryFile(path, device)) {
    ADD_FAILURE() << llvm::toString(std::move(error));
  }
  llvm::sys::fs::remove(path);
  return device;
}

} // namespace

TEST(DeviceTest, ConnectedPathDistances) {
  // A simple 4-qubit path: 0--1--2--3.
  Device device = Device::path(4);
  EXPECT_EQ(device.getNumQubits(), 4u);
  EXPECT_EQ(device.getDistance(Qubit(0), Qubit(0)), 0u);
  EXPECT_EQ(device.getDistance(Qubit(0), Qubit(3)), 3u);
  EXPECT_TRUE(device.areConnected(Qubit(1), Qubit(2)));
  EXPECT_FALSE(device.areConnected(Qubit(0), Qubit(2)));
  EXPECT_TRUE(device.hasPath(Qubit(0), Qubit(3)));
}

TEST(DeviceTest, DisconnectedComponentsAreUnreachable) {
  // Two islands: {0,1,2} (a line) and {3,4} (an edge).
  Device device = deviceFromTopology("Number of nodes: 5\n"
                                     "0 --> {1}\n"
                                     "1 --> {0, 2}\n"
                                     "2 --> {1}\n"
                                     "3 --> {4}\n"
                                     "4 --> {3}\n");
  ASSERT_EQ(device.getNumQubits(), 5u);

  // Within an island, distances and paths are finite.
  EXPECT_EQ(device.getDistance(Qubit(0), Qubit(2)), 2u);
  EXPECT_TRUE(device.hasPath(Qubit(0), Qubit(2)));
  EXPECT_EQ(device.getShortestPath(Qubit(0), Qubit(2)).size(), 3u);
  EXPECT_EQ(device.getDistance(Qubit(3), Qubit(4)), 1u);

  // Across islands, every pair is unreachable and every path is empty.
  EXPECT_EQ(device.getDistance(Qubit(2), Qubit(3)),
            Device::unreachableDistance);
  EXPECT_FALSE(device.hasPath(Qubit(0), Qubit(4)));
  EXPECT_TRUE(device.getShortestPath(Qubit(1), Qubit(3)).empty());
}

TEST(DeviceTest, IsolatedQubitReachesOnlyItself) {
  // Nodes 0 and 1 form an island; node 2 is declared but has no edges.
  Device device = deviceFromTopology("Number of nodes: 3\n"
                                     "0 --> {1}\n"
                                     "1 --> {0}\n");
  ASSERT_EQ(device.getNumQubits(), 3u);
  EXPECT_EQ(device.getDistance(Qubit(2), Qubit(2)), 0u);
  EXPECT_FALSE(device.hasPath(Qubit(0), Qubit(2)));
  EXPECT_EQ(device.getDistance(Qubit(1), Qubit(2)),
            Device::unreachableDistance);
}

TEST(DeviceTest, TryFileRejectsMissingFile) {
  Device device;
  llvm::Error error =
      Device::tryFile("/nonexistent/topology-does-not-exist.txt", device);
  EXPECT_TRUE(static_cast<bool>(error));
  llvm::consumeError(std::move(error));
}

TEST(DeviceTest, TryFileRejectsOutOfRangeQubit) {
  // Declares two nodes but references qubit index 5.
  llvm::SmallString<128> path;
  int fd = -1;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("topology", "txt", fd, path));
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    os << "Number of nodes: 2\n0 --> {5}\n";
  }
  Device device;
  llvm::Error error = Device::tryFile(path, device);
  EXPECT_TRUE(static_cast<bool>(error));
  llvm::consumeError(std::move(error));
  llvm::sys::fs::remove(path);
}
