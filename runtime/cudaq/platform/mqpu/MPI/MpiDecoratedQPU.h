/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform/qpu.h"
#include "cudaq/distributed/mpi_plugin.h"

namespace cudaq {
namespace details {

class MpiDecoratedQPU : public cudaq::QPU {
  cudaqDistributedCommunicator_t *qpuComm;
  cudaqDistributedCommunicator_t *worldComm;
  cudaqDistributedInterface_t *mpiInterface;
  cudaqDistributedGroup_t qpuGroup;
public:
  MpiDecoratedQPU() : QPU(){};
  MpiDecoratedQPU(std::size_t id) : QPU(id) {}
  MpiDecoratedQPU(const std::vector<int>& globalRanks);

  void enqueue(cudaq::QuantumTask &task) override;

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t, std::uint64_t,
               const std::vector<void *> &rawArgs) override;
  void
  configureExecutionContext(cudaq::ExecutionContext &context) const override;

  void beginExecution() override;

  void endExecution() override;

  void
  finalizeExecutionContext(cudaq::ExecutionContext &context) const override;

private:
  int getLocalMpiRank() const;
  int getGlobalMpiRank() const;
};
} // namespace details
} // namespace cudaq
