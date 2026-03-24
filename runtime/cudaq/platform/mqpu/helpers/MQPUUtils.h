/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/distributed/mpi_plugin.h"
#include <string>
#include <vector>

namespace cudaq {
// Helper struct to start a REST server (`cudaq-qpud`) instance on a random
// TCP/IP port. This will wait until the server is ready to serve incoming HTTP
// requests. The server process is terminated automatically by the destructor.
struct AutoLaunchRestServerProcess {
  AutoLaunchRestServerProcess(int seed_offset);
  ~AutoLaunchRestServerProcess();
  std::string getUrl() const;
  AutoLaunchRestServerProcess(const AutoLaunchRestServerProcess &) = delete;
  AutoLaunchRestServerProcess &
  operator=(const AutoLaunchRestServerProcess &) = delete;

private:
  int m_pid;
  std::string m_url;
  std::string m_libPathEnv;
};

// Helper to retrieve the number of GPU.
// It works with or without CUDA dependency.
// If CUDA is present, returns the actual number of GPU devices. Otherwise,
// returns 0.
int getCudaDeviceCount();
// Helper to retrieve the number of MPI processes in the current execution
// context. If not running in an MPI context, returns 1.
int getMPIProcessCount();

cudaqDistributedCommunicator_t *
splitCommunitor(const cudaqDistributedCommunicator_t *comm,
                const std::vector<std::vector<int>> &rankGroups);
} // namespace cudaq
