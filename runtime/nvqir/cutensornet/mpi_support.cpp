/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "cudaq/distributed/mpi_plugin.h"
#include "tensornet_utils.h"
#include <cassert>
#include <cutensornet.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
// Hook to query this shared lib file location at runtime.
extern "C" {
void getThisLibPath() { return; }
}

/// @brief Query the full path to the this lib.
static const char *getThisSharedLibFilePath() {
  static thread_local std::string LIB_PATH;
  if (LIB_PATH.empty()) {
    // Use dladdr query this .so file
    void *needle = (void *)(intptr_t)getThisLibPath;
    Dl_info DLInfo;
    int err = dladdr(needle, &DLInfo);
    if (err != 0) {
      char link_path[PATH_MAX];
      // If the filename is a symlink, we need to resolve and return the
      // location of the actual .so file.
      if (realpath(DLInfo.dli_fname, link_path))
        LIB_PATH = link_path;
    }
  }

  return LIB_PATH.c_str();
}

/// @brief Retrieve the MPI plugin comm interface
static cudaqDistributedInterface_t *getMpiPluginInterface() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  if (!mpiInterface)
    throw std::runtime_error("Invalid MPI distributed plugin encountered");
  return mpiInterface;
}

/// @brief Retrieve the MPI plugin (type-erased) comm pointer
static cudaqDistributedCommunicator_t *getMpiCommWrapper() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  if (!comm)
    throw std::runtime_error("Invalid MPI distributed plugin encountered");
  return comm;
}

/// @brief Retrieve the path to the plugin implementation
static std::string getMpiPluginFilePath() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");

  return mpiPlugin->getPluginPath();
}

void initCuTensornetComm(cutensornetHandle_t cutnHandle) {
  cudaqDistributedInterface_t *mpiInterface = getMpiPluginInterface();
  cudaqDistributedCommunicator_t *comm = getMpiCommWrapper();
  assert(mpiInterface && comm);
  cudaqDistributedCommunicator_t *dupComm = nullptr;
  const auto dupStatus = mpiInterface->CommDup(comm, &dupComm);
  if (dupStatus != 0 || dupComm == nullptr)
    throw std::runtime_error("Failed to duplicate the MPI communicator when "
                             "initializing cutensornet MPI");

  // If CUTENSORNET_COMM_LIB environment variable is not set,
  // use this builtin plugin shim (redirect MPI calls to CUDA-Q plugin)
  if (std::getenv("CUTENSORNET_COMM_LIB") == nullptr) {
    CUDAQ_INFO("Enabling cuTensorNet MPI without environment variable "
               "CUTENSORNET_COMM_LIB. \nUse the builtin cuTensorNet "
               "communicator lib from '{}' - CUDA-Q MPI plugin {}.",
               getThisSharedLibFilePath(), getMpiPluginFilePath());
    setenv("CUTENSORNET_COMM_LIB", getThisSharedLibFilePath(), 0);
  }

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

// Implementing cutensornet's COMM interface by delegating wrapped MPI calls to
// the underlying CUDA-Q MPI plugin. This will make this library
// compatible with CUTENSORNET_COMM_LIB API. Converts CUDA data type to the
// corresponding CUDA-Q shim type enum

/// Convert cutensornet CUDA datatype enum
static DataType convertCudaToMpiDataType(const cudaDataType_t cudaDataType) {
  switch (cudaDataType) {
  case CUDA_R_8I:
    return INT_8;
  case CUDA_R_16I:
    return INT_16;
  case CUDA_R_32I:
    return INT_32;
  case CUDA_R_64I:
    return INT_64;
  case CUDA_R_32F:
    return FLOAT_32;
  case CUDA_R_64F:
    return FLOAT_64;
  case CUDA_C_32F:
    return FLOAT_COMPLEX;
  case CUDA_C_64F:
    return DOUBLE_COMPLEX;
  default:
    throw std::runtime_error(
        "Unsupported data type encountered in cutensornet communicator plugin");
  }
  __builtin_unreachable();
}

/// Convert the type-erased Comm object
static cudaqDistributedCommunicator_t
convertMpiCommunicator(const cutensornetDistributedCommunicator_t *cutnComm) {
  cudaqDistributedCommunicator_t comm{cutnComm->commPtr, cutnComm->commSize};
  return comm;
}

#ifdef __cplusplus
extern "C" {
#endif
/// MPI_Comm_size wrapper
int cutensornetMpiCommSize(const cutensornetDistributedCommunicator_t *comm,
                           int32_t *numRanks) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->getNumRanks(&cudaqComm, numRanks);
}

/// Returns the size of the local subgroup of processes sharing node memory
int cutensornetMpiCommSizeShared(
    const cutensornetDistributedCommunicator_t *comm, int32_t *numRanks) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->getCommSizeShared(&cudaqComm, numRanks);
}

/// MPI_Comm_rank wrapper
int cutensornetMpiCommRank(const cutensornetDistributedCommunicator_t *comm,
                           int32_t *procRank) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->getProcRank(&cudaqComm, procRank);
}

/// MPI_Barrier wrapper
int cutensornetMpiBarrier(const cutensornetDistributedCommunicator_t *comm) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Barrier(&cudaqComm);
}

/// MPI_Bcast wrapper
int cutensornetMpiBcast(const cutensornetDistributedCommunicator_t *comm,
                        void *buffer, int32_t count, cudaDataType_t datatype,
                        int32_t root) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Bcast(
      &cudaqComm, buffer, count, convertCudaToMpiDataType(datatype), root);
}

/// MPI_Allreduce wrapper
int cutensornetMpiAllreduce(const cutensornetDistributedCommunicator_t *comm,
                            const void *bufferIn, void *bufferOut,
                            int32_t count, cudaDataType_t datatype) {
  ScopedTraceWithContext(__FUNCTION__);
  // cutensornet expects MPI_SUM in this API
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Allreduce(
      &cudaqComm, bufferIn, bufferOut, count,
      convertCudaToMpiDataType(datatype), SUM);
}

/// MPI_Allreduce IN_PLACE wrapper
int cutensornetMpiAllreduceInPlace(
    const cutensornetDistributedCommunicator_t *comm, void *buffer,
    int32_t count, cudaDataType_t datatype) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  // cutensornet expects MPI_SUM in this API
  return getMpiPluginInterface()->AllreduceInPlace(
      &cudaqComm, buffer, count, convertCudaToMpiDataType(datatype), SUM);
}

/// MPI_Allreduce IN_PLACE MIN wrapper
int cutensornetMpiAllreduceInPlaceMin(
    const cutensornetDistributedCommunicator_t *comm, void *buffer,
    int32_t count, cudaDataType_t datatype) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  // cutensornet expects MPI_SUM in this API
  return getMpiPluginInterface()->AllreduceInPlace(
      &cudaqComm, buffer, count, convertCudaToMpiDataType(datatype), MIN);
}

/// MPI_Allreduce DOUBLE_INT MINLOC wrapper
int cutensornetMpiAllreduceDoubleIntMinloc(
    const cutensornetDistributedCommunicator_t *comm,
    const void *bufferIn, // *struct {double; int;}
    void *bufferOut)      // *struct {double; int;}
{
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Allreduce(&cudaqComm, bufferIn, bufferOut, 1,
                                            FLOAT_64, MIN_LOC);
}

/// MPI_Allgather wrapper
int cutensornetMpiAllgather(const cutensornetDistributedCommunicator_t *comm,
                            const void *bufferIn, void *bufferOut,
                            int32_t count, cudaDataType_t datatype) {
  ScopedTraceWithContext(__FUNCTION__);
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Allgather(&cudaqComm, bufferIn, bufferOut,
                                            count,
                                            convertCudaToMpiDataType(datatype));
}

/// Distributed communication service API wrapper binding table (imported by
/// cuTensorNet). The exposed C symbol must be named as
/// "cutensornetCommInterface".
cutensornetDistributedInterface_t cutensornetCommInterface = {
    CUTENSORNET_DISTRIBUTED_INTERFACE_VERSION,
    cutensornetMpiCommSize,
    cutensornetMpiCommSizeShared,
    cutensornetMpiCommRank,
    cutensornetMpiBarrier,
    cutensornetMpiBcast,
    cutensornetMpiAllreduce,
    cutensornetMpiAllreduceInPlace,
    cutensornetMpiAllreduceInPlaceMin,
    cutensornetMpiAllreduceDoubleIntMinloc,
    cutensornetMpiAllgather};

#ifdef __cplusplus
} // extern "C"
#endif
