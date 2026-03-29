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
namespace details {

class QpuProcessGroup {
  cudaqDistributedCommunicator_t *qpuComm;
  cudaqDistributedGroup_t qpuGroup;
  std::vector<int> globalRanks;
public:
  QpuProcessGroup(const std::vector<int>& globalRanks);
  int getLocalMpiRank() const;
  bool contains(int globalRank) const;
  static int getGlobalMpiRank();
  static int getNumMpiRanks();
};
} // namespace details
} // namespace cudaq
