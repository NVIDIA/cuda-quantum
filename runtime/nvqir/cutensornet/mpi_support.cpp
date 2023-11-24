/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "tensornet_utils.h"

#if defined CUDAQ_HAS_MPI
#include <mpi.h>
#define HANDLE_MPI_ERROR(x)                                                    \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != MPI_SUCCESS) {                                                  \
      char error[MPI_MAX_ERROR_STRING];                                        \
      int len;                                                                 \
      MPI_Error_string(err, error, &len);                                      \
      printf("MPI Error: %s in line %d\n", error, __LINE__);                   \
      fflush(stdout);                                                          \
      MPI_Abort(MPI_COMM_WORLD, err);                                          \
    }                                                                          \
  };
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
  MPI_Comm cutnComm;
  // duplicate MPI communicator to dedicate it to cuTensorNet
  HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm));
  HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(
      cutnHandle, &cutnComm, sizeof(cutnComm)));
}

void resetCuTensornetComm(cutensornetHandle_t cutnHandle) {
  // Passing a nullptr to force a reset.
  HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(
      cutnHandle, nullptr, sizeof(MPI_Comm)));
}
#else
// Noop if we don't have MPI
void initCuTensornetComm(cutensornetHandle_t cutnHandle) {}
void resetCuTensornetComm(cutensornetHandle_t cutnHandle) {}
#endif