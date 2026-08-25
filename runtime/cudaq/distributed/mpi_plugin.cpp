/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mpi_plugin.h"
#include "common/PluginUtils.h"
#include <cassert>

namespace {
#define HANDLE_MPI_ERROR(x)                                                    \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != 0) {                                                            \
      printf("MPI Error encountered in line %d\n", __LINE__);                  \
      fflush(stdout);                                                          \
      std::abort();                                                            \
    }                                                                          \
  };
} // namespace

namespace cudaq {
bool MPIPlugin::isValidInterfaceLib(
    const std::string &distributedInterfaceLib) {
  const bool dlOpenOk =
      dlopen(distributedInterfaceLib.c_str(), RTLD_GLOBAL | RTLD_NOW);
  return dlOpenOk;
}

MPIPlugin::MPIPlugin(const std::string &distributedInterfaceLib) {
  m_libhandle = dlopen(distributedInterfaceLib.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (!m_libhandle) {
    const std::string errorMsg(dlerror());
    throw std::runtime_error("Unable to open distributed interface library '" +
                             distributedInterfaceLib + "': " + errorMsg);
  }
  m_distributedInterface = getUniquePluginInstance<cudaqDistributedInterface_t>(
      DISTRIBUTED_INTERFACE_GETTER_SYMBOL_NAME,
      distributedInterfaceLib.c_str());
  m_comm = getUniquePluginInstance<cudaqDistributedCommunicator_t>(
      COMM_GETTER_SYMBOL_NAME, distributedInterfaceLib.c_str());
  // getUniquePluginInstance should have thrown if cannot load.
  assert(m_distributedInterface && m_comm);
  m_valid = m_comm->commSize > 0;
  m_libFile = distributedInterfaceLib;
}

MPIPlugin::~MPIPlugin() {
  if (m_libhandle) {
    dlclose(m_libhandle);
    m_libhandle = nullptr;
  }
}

void MPIPlugin::initialize() {
  int argc{0};
  char **argv = nullptr;
  initialize(argc, argv);
}

void MPIPlugin::initialize(int argc, char **argv) {
  HANDLE_MPI_ERROR(m_distributedInterface->initialize(&argc, &argv));
}

int MPIPlugin::rank() {
  int pid{0};
  HANDLE_MPI_ERROR(m_distributedInterface->getProcRank(m_comm, &pid));
  return pid;
}

int MPIPlugin::num_ranks() {
  int np{0};
  HANDLE_MPI_ERROR(m_distributedInterface->getNumRanks(m_comm, &np));
  return np;
}

bool MPIPlugin::is_initialized() {
  int i{0};
  HANDLE_MPI_ERROR(m_distributedInterface->initialized(&i));
  return i == 1;
}

bool MPIPlugin::is_finalized() {
  int f{0};
  HANDLE_MPI_ERROR(m_distributedInterface->finalized(&f));
  return f == 1;
}

void MPIPlugin::all_gather(std::vector<double> &global,
                           const std::vector<double> &local) {
  HANDLE_MPI_ERROR(m_distributedInterface->Allgather(
      m_comm, local.data(), global.data(), local.size(), FLOAT_64));
}

void MPIPlugin::all_gather(std::vector<int> &global,
                           const std::vector<int> &local) {
  const auto dataType = (sizeof(int) == sizeof(int64_t)) ? INT_64 : INT_32;
  HANDLE_MPI_ERROR(m_distributedInterface->Allgather(
      m_comm, local.data(), global.data(), local.size(), dataType));
}

void MPIPlugin::broadcast(std::vector<double> &data, int rootRank) {
  HANDLE_MPI_ERROR(m_distributedInterface->Bcast(
      m_comm, data.data(), data.size(), FLOAT_64, rootRank));
}

void MPIPlugin::broadcast(std::string &data, int rootRank) {
  std::int32_t strLen = data.size();
  HANDLE_MPI_ERROR(
      m_distributedInterface->Bcast(m_comm, &strLen, 1, INT_32, rootRank));

  if (rank() != rootRank)
    data.resize(strLen);
  HANDLE_MPI_ERROR(m_distributedInterface->Bcast(
      m_comm, const_cast<char *>(data.data()), strLen, INT_8, rootRank));
}

void MPIPlugin::all_reduce(std::vector<double> &global,
                           const std::vector<double> &local, ReduceOp op) {
  HANDLE_MPI_ERROR(m_distributedInterface->Allreduce(
      m_comm, local.data(), global.data(), local.size(), FLOAT_64, op));
}

void MPIPlugin::finalize() {
  // Check if finalize has been called.
  int isFinalized{0};
  HANDLE_MPI_ERROR(m_distributedInterface->finalized(&isFinalized));

  if (isFinalized)
    return;

  if (rank() == 0)
    CUDAQ_INFO("Finalizing MPI.");

  // Finalize
  HANDLE_MPI_ERROR(m_distributedInterface->finalize());
}
} // namespace cudaq
