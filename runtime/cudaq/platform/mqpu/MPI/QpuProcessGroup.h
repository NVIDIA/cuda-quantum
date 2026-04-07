/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/distributed/mpi_plugin.h"

namespace cudaq {
class sample_result;
namespace details {

class QpuProcessGroup {
  cudaqDistributedCommunicator_t *qpuComm;
  cudaqDistributedGroup_t qpuGroup;
  std::vector<int> globalRanks;

public:
  QpuProcessGroup(const std::vector<int> &globalRanks);
  int getLocalMpiRank() const;
  bool contains(int globalRank) const;
  static void broadcast(cudaq::sample_result &data,
                        const QpuProcessGroup &rootGroup);
  static int getGlobalMpiRank();
  static int getNumMpiRanks();
  static bool isMpiInitialized();
  cudaqDistributedCommunicator_t *getCommunicator() const { return qpuComm; }
};
} // namespace details
} // namespace cudaq
