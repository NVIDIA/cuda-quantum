/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "cudaq/distributed/mpi_plugin.h"
#include <cassert>
#include <cudensitymat.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

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

// Implementing cudensitymat's COMM interface by delegating wrapped MPI calls to
// the underlying CUDA-Q MPI plugin. This will make this library
// compatible with CUDENSITYMAT_COMM_LIB API. Converts CUDA data type to the
// corresponding CUDA-Q shim type enum

/// Convert cudensitymat CUDA datatype enum
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
    throw std::runtime_error("Unsupported data type encountered in "
                             "cudensitymat communicator plugin");
  }
  __builtin_unreachable();
}

/// Convert the type-erased Comm object
static cudaqDistributedCommunicator_t
convertMpiCommunicator(const cudensitymatDistributedCommunicator_t *cutnComm) {
  cudaqDistributedCommunicator_t comm{cutnComm->commPtr, cutnComm->commSize};
  return comm;
}

#ifdef __cplusplus
extern "C" {
#endif

// Implementation of cudensitymat distributed interface by delegating to CUDA-Q
// MPI plugin API.
int cudensitymatMpiCommSize(const cudensitymatDistributedCommunicator_t *comm,
                            int32_t *numRanks) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->getNumRanks(&cudaqComm, numRanks);
}

int cudensitymatMpiCommSizeShared(
    const cudensitymatDistributedCommunicator_t *comm, int32_t *numRanks) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->getCommSizeShared(&cudaqComm, numRanks);
}

int cudensitymatMpiCommRank(const cudensitymatDistributedCommunicator_t *comm,
                            int32_t *procRank) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->getProcRank(&cudaqComm, procRank);
}

int cudensitymatMpiBarrier(const cudensitymatDistributedCommunicator_t *comm) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Barrier(&cudaqComm);
}

int cudensitymatMpiCreateRequest(cudensitymatDistributedRequest_t *request) {
  return getMpiPluginInterface()->CreateRequest(request);
}

int cudensitymatMpiDestroyRequest(cudensitymatDistributedRequest_t request) {
  return getMpiPluginInterface()->DestroyRequest(request);
}

int cudensitymatMpiWaitRequest(cudensitymatDistributedRequest_t request) {
  return getMpiPluginInterface()->WaitRequest(request);
}

int cudensitymatMpiTestRequest(cudensitymatDistributedRequest_t request,
                               int32_t *completed) {
  return getMpiPluginInterface()->TestRequest(request, completed);
}

int cudensitymatMpiSend(const cudensitymatDistributedCommunicator_t *comm,
                        const void *buffer, int32_t count,
                        cudaDataType_t datatype, int32_t destination,
                        int32_t tag) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Send(&cudaqComm, buffer, count,
                                       convertCudaToMpiDataType(datatype),
                                       destination, tag);
}

int cudensitymatMpiSendAsync(const cudensitymatDistributedCommunicator_t *comm,
                             const void *buffer, int32_t count,
                             cudaDataType_t datatype, int32_t destination,
                             int32_t tag,
                             cudensitymatDistributedRequest_t request) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->SendAsync(&cudaqComm, buffer, count,
                                            convertCudaToMpiDataType(datatype),
                                            destination, tag, request);
}

int cudensitymatMpiRecv(const cudensitymatDistributedCommunicator_t *comm,
                        void *buffer, int32_t count, cudaDataType_t datatype,
                        int32_t source, int32_t tag) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Recv(&cudaqComm, buffer, count,
                                       convertCudaToMpiDataType(datatype),
                                       source, tag);
}

int cudensitymatMpiRecvAsync(const cudensitymatDistributedCommunicator_t *comm,
                             void *buffer, int32_t count,
                             cudaDataType_t datatype, int32_t source,
                             int32_t tag,
                             cudensitymatDistributedRequest_t request) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->RecvAsync(&cudaqComm, buffer, count,
                                            convertCudaToMpiDataType(datatype),
                                            source, tag, request);
}

int cudensitymatMpiBcast(const cudensitymatDistributedCommunicator_t *comm,
                         void *buffer, int32_t count, cudaDataType_t datatype,
                         int32_t root) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Bcast(
      &cudaqComm, buffer, count, convertCudaToMpiDataType(datatype), root);
}

int cudensitymatMpiAllreduce(const cudensitymatDistributedCommunicator_t *comm,
                             const void *bufferIn, void *bufferOut,
                             int32_t count, cudaDataType_t datatype) {
  // cudensitymat expects MPI_SUM in this API
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Allreduce(
      &cudaqComm, bufferIn, bufferOut, count,
      convertCudaToMpiDataType(datatype), SUM);
}

int cudensitymatMpiAllreduceInPlace(
    const cudensitymatDistributedCommunicator_t *comm, void *buffer,
    int32_t count, cudaDataType_t datatype) {
  auto cudaqComm = convertMpiCommunicator(comm);
  // cudensitymat expects MPI_SUM in this API
  return getMpiPluginInterface()->AllreduceInPlace(
      &cudaqComm, buffer, count, convertCudaToMpiDataType(datatype), SUM);
}

int cudensitymatMpiAllreduceInPlaceMin(
    const cudensitymatDistributedCommunicator_t *comm, void *buffer,
    int32_t count, cudaDataType_t datatype) {
  auto cudaqComm = convertMpiCommunicator(comm);
  // cudensitymat expects MPI_SUM in this API
  return getMpiPluginInterface()->AllreduceInPlace(
      &cudaqComm, buffer, count, convertCudaToMpiDataType(datatype), MIN);
}

int cudensitymatMpiAllreduceDoubleIntMinloc(
    const cudensitymatDistributedCommunicator_t *comm, const void *bufferIn,
    void *bufferOut) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Allreduce(&cudaqComm, bufferIn, bufferOut, 1,
                                            FLOAT_64, MIN_LOC);
}

int cudensitymatMpiAllgather(const cudensitymatDistributedCommunicator_t *comm,
                             const void *bufferIn, void *bufferOut,
                             int32_t count, cudaDataType_t datatype) {
  auto cudaqComm = convertMpiCommunicator(comm);
  return getMpiPluginInterface()->Allgather(&cudaqComm, bufferIn, bufferOut,
                                            count,
                                            convertCudaToMpiDataType(datatype));
}

cudensitymatDistributedInterface_t cudensitymatCommInterface = {
    CUDENSITYMAT_DISTRIBUTED_INTERFACE_VERSION,
    cudensitymatMpiCommSize,
    cudensitymatMpiCommSizeShared,
    cudensitymatMpiCommRank,
    cudensitymatMpiBarrier,
    cudensitymatMpiCreateRequest,
    cudensitymatMpiDestroyRequest,
    cudensitymatMpiWaitRequest,
    cudensitymatMpiTestRequest,
    cudensitymatMpiSend,
    cudensitymatMpiSendAsync,
    cudensitymatMpiRecv,
    cudensitymatMpiRecvAsync,
    cudensitymatMpiBcast,
    cudensitymatMpiAllreduce,
    cudensitymatMpiAllreduceInPlace,
    cudensitymatMpiAllreduceInPlaceMin,
    cudensitymatMpiAllreduceDoubleIntMinloc,
    cudensitymatMpiAllgather};

#ifdef __cplusplus
} // extern "C"
#endif
