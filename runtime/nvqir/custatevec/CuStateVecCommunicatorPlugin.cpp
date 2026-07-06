/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/distributed/mpi_plugin.h"

#include <custatevecEx_ext.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace {

/// Adapts a CUDA-Q distributed communicator to the `cuStateVecEx` plugin ABI.
struct CudaqCommunicator : custatevecExCommunicator_t {
  cudaqDistributedInterface_t *distributed = nullptr;
  cudaqDistributedCommunicator_t communicator{};
};

std::mutex initializationMutex;
std::mutex communicatorMutex;
cudaqDistributedCommunicator_t selectedCommunicator{};
bool communicatorWasSelected = false;
bool initializedByBridge = false;

cudaq::MPIPlugin &plugin() {
  auto *const value = cudaq::mpi::getMpiPlugin();
  if (!value || !value->get() || !value->getComm())
    throw std::runtime_error("CUDA-Q distributed plugin is unavailable.");
  return *value;
}

DataType convertType(cudaDataType_t type) {
  switch (type) {
  case CUDA_R_8U:
    return UINT_8;
  case CUDA_R_8I:
    return INT_8;
  case CUDA_R_16U:
    return UINT_16;
  case CUDA_R_16I:
    return INT_16;
  case CUDA_R_32U:
    return UINT_32;
  case CUDA_R_32I:
    return INT_32;
  case CUDA_R_64U:
    return UINT_64;
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
    throw std::invalid_argument("Unsupported communicator data type.");
  }
}

custatevecExCommunicatorStatus_t status(int value) {
  return static_cast<custatevecExCommunicatorStatus_t>(value);
}

CudaqCommunicator &asCudaq(custatevecExCommunicator_t *communicator) {
  return *static_cast<CudaqCommunicator *>(communicator);
}

// --- custatevecEx communicator ABI implementation ---------------------------
// The functions below implement the custatevecExCommunicatorModule_t and
// custatevecExCommunicatorInterface_t vtables. cuStateVecEx invokes them
// through those function pointers, so each signature is fixed by the
// custatevecEx typedefs.
custatevecExCommunicatorStatus_t getVersion(int32_t *major, int32_t *minor) {
  if (!major || !minor)
    return status(1);
  *major = 1;
  *minor = 0;
  return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
}

custatevecExCommunicatorStatus_t initialize(void *, int *argc, char ***argv) {
  try {
    std::scoped_lock lock(initializationMutex);
    auto &mpi = plugin();
    int32_t initialized = 0;
    const int queryStatus = mpi.get()->initialized(&initialized);
    if (queryStatus != 0)
      return status(queryStatus);
    if (!initialized) {
      const int initStatus =
          mpi.get()->initialize(reinterpret_cast<int32_t *>(argc), argv);
      if (initStatus != 0)
        return status(initStatus);
      initializedByBridge = true;
    }
    return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t initialized(void *, int *flag) {
  try {
    return status(
        plugin().get()->initialized(reinterpret_cast<int32_t *>(flag)));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t finalize(void *) {
  try {
    std::scoped_lock lock(initializationMutex);
    if (!initializedByBridge)
      return CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
    const int finalizeStatus = plugin().get()->finalize();
    if (finalizeStatus == 0)
      initializedByBridge = false;
    return status(finalizeStatus);
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t finalized(void *, int *flag) {
  try {
    return status(plugin().get()->finalized(reinterpret_cast<int32_t *>(flag)));
  } catch (...) {
    return status(1);
  }
}

cudaqDistributedCommunicator_t activeCommunicator() {
  std::scoped_lock lock(communicatorMutex);
  if (communicatorWasSelected)
    return selectedCommunicator;
  return *plugin().getComm();
}

custatevecExCommunicatorStatus_t getSizeAndRank(void *, int32_t *size,
                                                int32_t *rank) {
  try {
    auto &mpi = plugin();
    const auto communicator = activeCommunicator();
    const int sizeStatus = mpi.get()->getNumRanks(&communicator, size);
    if (sizeStatus != 0)
      return status(sizeStatus);
    return status(mpi.get()->getProcRank(&communicator, rank));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t abort(custatevecExCommunicator_t *value,
                                       int error) {
  auto &comm = asCudaq(value);
  return status(comm.distributed->Abort(&comm.communicator, error));
}

custatevecExCommunicatorStatus_t getSize(custatevecExCommunicator_t *value,
                                         int *size) {
  auto &comm = asCudaq(value);
  return status(comm.distributed->getNumRanks(
      &comm.communicator, reinterpret_cast<int32_t *>(size)));
}

custatevecExCommunicatorStatus_t getRank(custatevecExCommunicator_t *value,
                                         int *rank) {
  auto &comm = asCudaq(value);
  return status(comm.distributed->getProcRank(
      &comm.communicator, reinterpret_cast<int32_t *>(rank)));
}

custatevecExCommunicatorStatus_t barrier(custatevecExCommunicator_t *value) {
  auto &comm = asCudaq(value);
  return status(comm.distributed->Barrier(&comm.communicator));
}

custatevecExCommunicatorStatus_t bcast(custatevecExCommunicator_t *value,
                                       void *buffer, int count,
                                       cudaDataType_t type, int root) {
  auto &comm = asCudaq(value);
  try {
    return status(comm.distributed->Bcast(&comm.communicator, buffer, count,
                                          convertType(type), root));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t allreduce(custatevecExCommunicator_t *value,
                                           const void *sendBuffer,
                                           void *receiveBuffer, int count,
                                           cudaDataType_t type) {
  auto &comm = asCudaq(value);
  try {
    if (sendBuffer == receiveBuffer)
      return status(comm.distributed->AllreduceInPlace(
          &comm.communicator, receiveBuffer, count, convertType(type), SUM));

    return status(comm.distributed->Allreduce(&comm.communicator, sendBuffer,
                                              receiveBuffer, count,
                                              convertType(type), SUM));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t allgather(custatevecExCommunicator_t *value,
                                           const void *sendBuffer,
                                           void *receiveBuffer, int count,
                                           cudaDataType_t type) {
  auto &comm = asCudaq(value);
  try {
    return status(comm.distributed->Allgather(&comm.communicator, sendBuffer,
                                              receiveBuffer, count,
                                              convertType(type)));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t
allgatherv(custatevecExCommunicator_t *value, const void *sendBuffer,
           int sendCount, void *receiveBuffer, const int *receiveCounts,
           const int *displacements, cudaDataType_t type) {
  auto &comm = asCudaq(value);
  try {
    return status(comm.distributed->AllgatherV(
        &comm.communicator, sendBuffer, sendCount, receiveBuffer, receiveCounts,
        displacements, convertType(type)));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t sendAsync(custatevecExCommunicator_t *value,
                                           const void *buffer, int count,
                                           cudaDataType_t type, int peer,
                                           int32_t tag) {
  auto &comm = asCudaq(value);
  try {
    return status(comm.distributed->SendAsync(&comm.communicator, buffer, count,
                                              convertType(type), peer, tag,
                                              nullptr));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t recvAsync(custatevecExCommunicator_t *value,
                                           void *buffer, int count,
                                           cudaDataType_t type, int peer,
                                           int32_t tag) {
  auto &comm = asCudaq(value);
  try {
    return status(comm.distributed->RecvAsync(&comm.communicator, buffer, count,
                                              convertType(type), peer, tag,
                                              nullptr));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t
sendRecvAsync(custatevecExCommunicator_t *value, const void *sendBuffer,
              void *receiveBuffer, int count, cudaDataType_t type, int peer,
              int32_t tag) {
  auto &comm = asCudaq(value);
  try {
    return status(comm.distributed->SendRecvAsync(
        &comm.communicator, sendBuffer, receiveBuffer, count, convertType(type),
        peer, tag));
  } catch (...) {
    return status(1);
  }
}

custatevecExCommunicatorStatus_t
synchronize(custatevecExCommunicator_t *value) {
  auto &comm = asCudaq(value);
  return status(comm.distributed->Synchronize(&comm.communicator));
}

// custatevecEx ABI vtable (point-to-point / collective ops). The member order
// must match custatevecExCommunicatorInterface_t exactly.
const custatevecExCommunicatorInterface_t communicatorInterface = {
    abort,      getSize,   getRank,   barrier,       bcast,       allgather,
    allgatherv, sendAsync, recvAsync, sendRecvAsync, synchronize, allreduce};

custatevecExCommunicator_t *createCommunicator(void *) {
  try {
    auto result = std::make_unique<CudaqCommunicator>();
    result->intf = &communicatorInterface;
    result->distributed = plugin().get();
    result->communicator = activeCommunicator();
    return result.release();
  } catch (...) {
    return nullptr;
  }
}

void destroyCommunicator(void *, custatevecExCommunicator_t *communicator) {
  delete static_cast<CudaqCommunicator *>(communicator);
}

// custatevecEx ABI vtable (module lifecycle). The member order must match
// custatevecExCommunicatorModule_t exactly.
const custatevecExCommunicatorModule_t communicatorModule = {
    getVersion, initialize,     initialized,        finalize,
    finalized,  getSizeAndRank, createCommunicator, destroyCommunicator};

} // namespace

// ABI symbol for `CuStateVecCommunicator::setCommunicator`
extern "C" custatevecStatus_t
cudaqCustatevecExSetCommunicator(void *communicator,
                                 std::size_t communicatorSize) {
  if (!communicator || communicatorSize == 0)
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  std::scoped_lock lock(communicatorMutex);
  selectedCommunicator = {communicator, communicatorSize};
  communicatorWasSelected = true;
  return CUSTATEVEC_STATUS_SUCCESS;
}

// ABI symbol resolved by `CuStateVecCommunicator::setCommunicator`
extern "C" custatevecStatus_t
cudaqCustatevecExGetCommunicator(void **communicator,
                                 std::size_t *communicatorSize) {
  if (!communicator || !communicatorSize)
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  try {
    const auto active = activeCommunicator();
    *communicator = active.commPtr;
    *communicatorSize = active.commSize;
    return CUSTATEVEC_STATUS_SUCCESS;
  } catch (...) {
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  }
}

// Fixed-name ABI entry point cuStateVecEx resolves to obtain the communicator
// module.
extern "C" custatevecStatus_t custatevecExCommunicatorGetModuleEXT(
    const custatevecExCommunicatorModule_t **module) {
  if (!module)
    return CUSTATEVEC_STATUS_INVALID_VALUE;
  *module = &communicatorModule;
  return CUSTATEVEC_STATUS_SUCCESS;
}
