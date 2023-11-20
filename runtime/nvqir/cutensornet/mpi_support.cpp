/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/distributed/mpi_plugin.h"
#include "tensornet_utils.h"
namespace cudaq::mpi {
cudaq::MPIPlugin *getMpiPlugin(bool unsafe = false);
} // namespace cudaq::mpi

void initCuTensornetComm(cutensornetHandle_t cutnHandle) {
  // If the CUTENSORNET_COMM_LIB environment variable is not set, print a
  // warning message since cutensornet will likely to fail.
  //
  // Note: this initialization only happens when the user initializes
  // MPI explicitly. In this case, the user will need to define the environment
  // variable CUTENSORNET_COMM_LIB as described in the Getting Started section
  // of the cuTensorNet library documentation (Installation and Compilation).
  if (std::getenv("CUTENSORNET_COMM_LIB") == nullptr)
    printf("[Warning] Enabling cuTensorNet MPI without environment variable "
           "CUTENSORNET_COMM_LIB.\nMPI parallelization inside cuTensorNet "
           "library may cause an error.\n");

  // duplicate MPI communicator to dedicate it to cuTensorNet
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Invalid MPI distributed plugin encountered when "
                             "initializing cutensornet MPI");
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  if (!mpiInterface || !comm)
    throw std::runtime_error("Invalid MPI distributed plugin encountered when "
                             "initializing cutensornet MPI");
  cudaqDistributedCommunicator_t *dupComm = nullptr;
  const auto dupStatus = mpiInterface->CommDup(comm, &dupComm);
  if (dupStatus != 0 || dupComm == nullptr)
    throw std::runtime_error("Failed to duplicate the MPI communicator when "
                             "initializing cutensornet MPI");

  HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(
      cutnHandle, dupComm->commPtr, dupComm->commSize));
}

void resetCuTensornetComm(cutensornetHandle_t cutnHandle) {
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Invalid MPI distributed plugin encountered when "
                             "initializing cutensornet MPI");
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  if (!mpiInterface || !comm)
    throw std::runtime_error("Invalid MPI distributed plugin encountered when "
                             "initializing cutensornet MPI");
  // Passing a nullptr to force a reset.
  HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(
      cutnHandle, nullptr, comm->commSize));
}