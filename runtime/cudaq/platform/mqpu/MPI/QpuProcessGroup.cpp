/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QpuProcessGroup.h"
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

namespace {
static cudaq::MPIPlugin *g_mpiPlugin = nullptr;
static cudaqDistributedCommunicator_t *worldComm = nullptr;
static cudaqDistributedInterface_t *mpiInterface = nullptr;
static const bool initialized = []() {
  g_mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!g_mpiPlugin) {
    throw std::runtime_error("Failed to retrieve MPI plugin");
  }
  mpiInterface = g_mpiPlugin->get();
  worldComm = g_mpiPlugin->getComm();
  return true;
}();
} // namespace

namespace cudaq {
namespace details {

QpuProcessGroup::QpuProcessGroup(const std::vector<int> &globalRanks)
    : globalRanks(globalRanks) {
  int rank = 0;
  HANDLE_MPI_ERROR(mpiInterface->getProcRank(worldComm, &rank));

  {
    volatile int i = 5;
    // ENV variable to control whether to wait for attach
    const char *envVal = std::getenv("CUDAQ_MQPU_WAIT_FOR_ATTACH");
    if (envVal != nullptr) {
      i = 0;
    }
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Rank %d PID %d on %s ready for attach\n", rank, getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(5);
  }

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
            << " Creating QpuProcessGroup with global ranks: ";
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
}

int QpuProcessGroup::getLocalMpiRank() const {
  int rank = 0;
  HANDLE_MPI_ERROR(mpiInterface->getProcRank(qpuComm, &rank));
  return rank;
}

int QpuProcessGroup::getGlobalMpiRank() {
  int initialized = 0;
  HANDLE_MPI_ERROR(mpiInterface->initialized(&initialized));
  if (!initialized)
    return 0;

  int rank = 0;
  HANDLE_MPI_ERROR(mpiInterface->getProcRank(worldComm, &rank));
  return rank;
}

int QpuProcessGroup::getNumMpiRanks() {
  int initialized = 0;
  HANDLE_MPI_ERROR(mpiInterface->initialized(&initialized));
  if (!initialized)
    return 1;

  int size = 0;
  HANDLE_MPI_ERROR(mpiInterface->getNumRanks(worldComm, &size));
  return size;
}

bool QpuProcessGroup::isMpiInitialized() {
  int initialized = 0;
  HANDLE_MPI_ERROR(mpiInterface->initialized(&initialized));
  return initialized != 0;
}

bool QpuProcessGroup::contains(int globalRank) const {
  return std::find(globalRanks.begin(), globalRanks.end(), globalRank) !=
         globalRanks.end();
}

void QpuProcessGroup::broadcast(cudaq::sample_result &data, const QpuProcessGroup &rootGroup) {
  const bool isInGroup = rootGroup.contains(getGlobalMpiRank());
  std::vector<std::size_t> serializedData;
  const auto sourceRank = rootGroup.globalRanks.front();
  if (isInGroup) {
    serializedData = data.serialize();
    int size = serializedData.size();
    HANDLE_MPI_ERROR(
        mpiInterface->Bcast(worldComm, &size, 1, INT_32, sourceRank));
    HANDLE_MPI_ERROR(mpiInterface->Bcast(worldComm, serializedData.data(), size,
                                         INT_64, sourceRank));
  } else {
    int size = 0;
    HANDLE_MPI_ERROR(mpiInterface->Bcast(worldComm, &size, 1, INT_32, sourceRank));
    serializedData.resize(size);
    HANDLE_MPI_ERROR(
        mpiInterface->Bcast(worldComm, serializedData.data(), size, INT_64, sourceRank));
    data.deserialize(serializedData);
  }
}

} // namespace details
} // namespace cudaq
