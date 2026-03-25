/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MpiDecoratedQPU.h"
#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"

#define HANDLE_MPI_ERROR(x)                                                    \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != 0) {                                                            \
      printf("MPI Error encountered in line %d\n", __LINE__);                  \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  };

namespace cudaq {
namespace details {

MpiDecoratedQPU::MpiDecoratedQPU(const std::vector<int> &globalRanks)
    : cudaq::QPU() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  mpiInterface = mpiPlugin->get();
  worldComm = mpiPlugin->getComm();
  int rank = 0;
  HANDLE_MPI_ERROR(mpiInterface->getProcRank(worldComm, &rank));
  if (std::find(globalRanks.begin(), globalRanks.end(), rank) ==
      globalRanks.end()) {
    // This rank is not part of this QPU's communicator group, so we can skip
    // the communicator splitting and just return. The QPU will not be
    // functional on this rank.
    qpuComm = nullptr;
    return;
  }
  // Split the global communicator into sub-communicators based on the provided
  // global ranks for this QPU.
  cudaqDistributedGroup_t world_group;
  HANDLE_MPI_ERROR(mpiInterface->CommGroup(worldComm, &world_group));
  std::cout << "Rank " << rank
            << " Creating MpiDecoratedQPU with global ranks: ";
  for (int rank : globalRanks) {
    std::cout << rank << " ";
  }
  std::cout << std::endl;
  int num_ranks = 0;
  HANDLE_MPI_ERROR(mpiInterface->getNumRanks(worldComm, &num_ranks));
  std::cout << "Rank " << rank
            << " Total number of ranks in world communicator: " << num_ranks
            << std::endl;

  HANDLE_MPI_ERROR(mpiInterface->GroupIncl(world_group, globalRanks.size(),
                                           globalRanks.data(), &qpuGroup));
  HANDLE_MPI_ERROR(
      mpiInterface->CommCreateGroup(worldComm, qpuGroup, 0, &qpuComm));

  // {
  //   volatile int i = 0;
  //   char hostname[256];
  //   gethostname(hostname, sizeof(hostname));
  //   printf("Rank %d PID %d on %s ready for attach\n", rank,
  //          getpid(), hostname);
  //   fflush(stdout);
  //   while (0 == i)
  //     sleep(5);
  // }
}

int MpiDecoratedQPU::getLocalMpiRank() const {
  int rank = 0;
  HANDLE_MPI_ERROR(mpiInterface->getProcRank(qpuComm, &rank));
  return rank;
}

int MpiDecoratedQPU::getGlobalMpiRank() const {
  int rank = 0;
  HANDLE_MPI_ERROR(mpiInterface->getProcRank(worldComm, &rank));
  return rank;
}

void MpiDecoratedQPU::enqueue(cudaq::QuantumTask &task) {
  // Note: enqueue is executed on the main thread, not the QPU execution
  // thread. Hence, do not set the CUDA device here.
  CUDAQ_INFO("Enqueue Task on QPU {}, MPI rank {} (global rank {})", qpu_id,
             getLocalMpiRank(), getGlobalMpiRank());

  execution_queue->enqueue(task);
}

cudaq::KernelThunkResultType MpiDecoratedQPU::launchKernel(
    const std::string &name, cudaq::KernelThunkType kernelFunc, void *args,
    std::uint64_t, std::uint64_t, const std::vector<void *> &rawArgs) {
  if (qpuComm == nullptr) {
    return cudaq::KernelThunkResultType{nullptr, 0};
  }
  CUDAQ_INFO("QPU::launchKernel GPU {}, MPI rank {} (global rank {})", qpu_id,
             getLocalMpiRank(), getGlobalMpiRank());

  return kernelFunc(args, /*differentMemorySpace=*/false);
}

void MpiDecoratedQPU::configureExecutionContext(
    cudaq::ExecutionContext &context) const {
  if (qpuComm == nullptr) {
    return;
  }
  CUDAQ_INFO("MultiQPUPlatform::configureExecutionContext QPU {}, MPI rank {} "
             "(global rank {})",
             qpu_id, getLocalMpiRank(), getGlobalMpiRank());
  if (noiseModel)
    context.noiseModel = noiseModel;

  context.executionManager = cudaq::getDefaultExecutionManager();
  context.executionManager->configureExecutionContext(context);
}

void MpiDecoratedQPU::beginExecution() {
  if (qpuComm == nullptr) {
    return;
  }
  cudaq::getExecutionContext()->executionManager->beginExecution();
}

void MpiDecoratedQPU::endExecution() {
  if (qpuComm == nullptr) {
    return;
  }
  cudaq::getExecutionContext()->executionManager->endExecution();
}

void MpiDecoratedQPU::finalizeExecutionContext(
    cudaq::ExecutionContext &context) const {
  if (qpuComm == nullptr) {
    return;
  }
  CUDAQ_INFO("MultiQPUPlatform::finalizeExecutionContext QPU {}, MPI rank {} "
             "(global rank {})",
             qpu_id, getLocalMpiRank(), getGlobalMpiRank());

  handleObservation(context);

  cudaq::getExecutionContext()->executionManager->finalizeExecutionContext(
      context);
}
} // namespace details
} // namespace cudaq
