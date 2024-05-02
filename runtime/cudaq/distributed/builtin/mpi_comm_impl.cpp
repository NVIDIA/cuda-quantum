/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*! \file mpi_comm_impl.cpp
    \brief Reference implementation of CUDA-Q MPI interface wrapper

    This is an implementation of the MPI shim interface defined in
   distributed_capi.h. This can be compiled and linked against an MPI
   implementation (e.g., OpenMPI or MPICH) to produce a runtime loadable plugin
   providing CUDA-Q with necessary MPI functionalities.
*/

#include "distributed_capi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// This MPI plugin does not use the [deprecated] C++ binding of MPI at ALL. The
// following flag makes sure the C++ bindings are not included.
#if !defined(MPICH_SKIP_MPICXX)
#define MPICH_SKIP_MPICXX 1
#endif
#if !defined(OMPI_SKIP_MPICXX)
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

namespace {
bool initCalledByThis = false;
MPI_Datatype convertType(DataType dataType) {
  switch (dataType) {
  case INT_8:
    return MPI_INT8_T;
  case INT_16:
    return MPI_INT16_T;
  case INT_32:
    return MPI_INT32_T;
  case INT_64:
    return MPI_INT64_T;
  case FLOAT_32:
    return MPI_FLOAT;
  case FLOAT_64:
    return MPI_DOUBLE;
  case FLOAT_COMPLEX:
    return MPI_C_FLOAT_COMPLEX;
  case DOUBLE_COMPLEX:
    return MPI_C_DOUBLE_COMPLEX;
  }
  __builtin_unreachable();
}

MPI_Datatype convertTypeMinLoc(DataType dataType) {
  switch (dataType) {
  case FLOAT_32:
    return MPI_FLOAT_INT;
  case FLOAT_64:
    return MPI_DOUBLE_INT;
  default:
    throw std::runtime_error("Unsupported MINLOC data type");
  }
  __builtin_unreachable();
}

MPI_Op convertType(ReduceOp opType) {
  switch (opType) {
  case SUM:
    return MPI_SUM;
  case PROD:
    return MPI_PROD;
  case MIN:
    return MPI_MIN;
  case MIN_LOC:
    return MPI_MINLOC;
  }
  __builtin_unreachable();
}

MPI_Comm unpackMpiCommunicator(const cudaqDistributedCommunicator_t *comm) {
  if (comm->commPtr == NULL)
    return MPI_COMM_NULL;
  if (sizeof(MPI_Comm) != comm->commSize) {
    printf("#FATAL: MPI_Comm object has unexpected size!\n");
    exit(EXIT_FAILURE);
  }
  return *((MPI_Comm *)(comm->commPtr));
}

/// @brief Tracking in-flight non-blocking send and receive requests.
struct PendingRequest {
  MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  int nActiveRequests;
  PendingRequest() : nActiveRequests(0){};
  static std::mutex g_mutex;
  static std::unordered_map<const cudaqDistributedCommunicator_t *,
                            PendingRequest>
      g_requests;
};

std::mutex PendingRequest::g_mutex;
std::unordered_map<const cudaqDistributedCommunicator_t *, PendingRequest>
    PendingRequest::g_requests;
} // namespace
extern "C" {

/// @brief Wrapper of MPI_Init
static int mpi_initialize(int32_t *argc, char ***argv) {
  int flag = 0;
  int res = MPI_Initialized(&flag);
  if (res != MPI_SUCCESS)
    return res;
  // This has been initialized, nothing to do.
  if (flag)
    return MPI_SUCCESS;
  initCalledByThis = true;
  return MPI_Init(argc, argv);
}

/// @brief Wrapper of MPI_Finalize
static int mpi_finalize() {
  if (!initCalledByThis)
    return MPI_SUCCESS;
  return MPI_Finalize();
}

/// @brief Wrapper of MPI_Initialized
static int mpi_initialized(int32_t *flag) { return MPI_Initialized(flag); }

/// @brief Wrapper of MPI_Finalized
static int mpi_finalized(int32_t *flag) { return MPI_Finalized(flag); }

/// @brief Wrapper of MPI_Comm_size
static int mpi_getNumRanks(const cudaqDistributedCommunicator_t *comm,
                           int32_t *size) {
  return MPI_Comm_size(unpackMpiCommunicator(comm), size);
}

/// @brief Wrapper of MPI_Comm_rank
static int mpi_getProcRank(const cudaqDistributedCommunicator_t *comm,
                           int32_t *rank) {
  return MPI_Comm_rank(unpackMpiCommunicator(comm), rank);
}

/// @brief Returns the size of the local subgroup of processes sharing node
/// memory
static int mpi_getCommSizeShared(const cudaqDistributedCommunicator_t *comm,
                                 int32_t *numRanks) {
  *numRanks = 0;
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "mpi_hw_resource_type", "mpi_shared_memory");
  int procRank = -1;
  int mpiErr = MPI_Comm_rank(unpackMpiCommunicator(comm), &procRank);
  if (mpiErr == MPI_SUCCESS) {
    MPI_Comm localComm;
    mpiErr =
        MPI_Comm_split_type(unpackMpiCommunicator(comm), MPI_COMM_TYPE_SHARED,
                            procRank, info, &localComm);
    if (mpiErr == MPI_SUCCESS) {
      int nranks = 0;
      mpiErr = MPI_Comm_size(localComm, &nranks);
      *numRanks = nranks;
      MPI_Comm_free(&localComm);
    }
  }
  return mpiErr;
}

/// @brief Wrapper of MPI_Barrier
static int mpi_Barrier(const cudaqDistributedCommunicator_t *comm) {
  return MPI_Barrier(unpackMpiCommunicator(comm));
}

/// @brief Wrapper of MPI_Bcast
static int mpi_Bcast(const cudaqDistributedCommunicator_t *comm, void *buffer,
                     int32_t count, DataType dataType, int32_t rootRank) {
  return MPI_Bcast(buffer, count, convertType(dataType), rootRank,
                   unpackMpiCommunicator(comm));
}

/// @brief Wrapper of MPI_Allreduce
static int mpi_Allreduce(const cudaqDistributedCommunicator_t *comm,
                         const void *sendBuffer, void *recvBuffer,
                         int32_t count, DataType dataType, ReduceOp opType) {
  if (opType == MIN_LOC) {
    return MPI_Allreduce(sendBuffer, recvBuffer, count,
                         convertTypeMinLoc(dataType), convertType(opType),
                         unpackMpiCommunicator(comm));
  } else {
    return MPI_Allreduce(sendBuffer, recvBuffer, count, convertType(dataType),
                         convertType(opType), unpackMpiCommunicator(comm));
  }
}

/// @brief Wrapper of MPI_Allreduce with MPI_IN_PLACE
static int mpi_AllreduceInplace(const cudaqDistributedCommunicator_t *comm,
                                void *recvBuffer, int32_t count,
                                DataType dataType, ReduceOp opType) {
  return MPI_Allreduce(MPI_IN_PLACE, recvBuffer, count, convertType(dataType),
                       convertType(opType), unpackMpiCommunicator(comm));
}

/// @brief Wrapper of MPI_Allgather
static int mpi_Allgather(const cudaqDistributedCommunicator_t *comm,
                         const void *sendBuffer, void *recvBuffer,
                         int32_t count, DataType dataType) {
  return MPI_Allgather(sendBuffer, count, convertType(dataType), recvBuffer,
                       count, convertType(dataType),
                       unpackMpiCommunicator(comm));
}

/// @brief Wrapper of MPI_Allgatherv
static int mpi_AllgatherV(const cudaqDistributedCommunicator_t *comm,
                          const void *sendBuf, int sendCount, void *recvBuf,
                          const int *recvCounts, const int *displs,
                          DataType dataType) {
  return MPI_Allgatherv(sendBuf, sendCount, convertType(dataType), recvBuf,
                        recvCounts, displs, convertType(dataType),
                        unpackMpiCommunicator(comm));
}

/// @brief Wrapper of MPI_Isend and track pending requests for synchronization
static int mpi_SendAsync(const cudaqDistributedCommunicator_t *comm,
                         const void *buf, int count, DataType dataType,
                         int peer, int32_t tag) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  if (PendingRequest::g_requests[comm].nActiveRequests == 2)
    return -1;
  MPI_Request *request =
      &(PendingRequest::g_requests[comm]
            .requests[PendingRequest::g_requests[comm].nActiveRequests]);
  int res = MPI_Isend(buf, count, convertType(dataType), peer, tag,
                      unpackMpiCommunicator(comm), request);
  if (res != MPI_SUCCESS) {
    MPI_Cancel(request);
    return res;
  }
  ++PendingRequest::g_requests[comm].nActiveRequests;
  return 0;
}

/// @brief Wrapper of MPI_Irecv and track pending requests for synchronization
static int mpi_RecvAsync(const cudaqDistributedCommunicator_t *comm, void *buf,
                         int count, DataType dataType, int peer, int32_t tag) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  if (PendingRequest::g_requests[comm].nActiveRequests == 2)
    return -1;
  MPI_Request *request =
      &(PendingRequest::g_requests[comm]
            .requests[PendingRequest::g_requests[comm].nActiveRequests]);
  int res = MPI_Irecv(buf, count, convertType(dataType), peer, tag,
                      unpackMpiCommunicator(comm), request);
  if (res != MPI_SUCCESS) {
    MPI_Cancel(request);
    return res;
  }
  ++PendingRequest::g_requests[comm].nActiveRequests;
  return 0;
}

/// @brief Combined MPI_Isend and MPI_Irecv requests
static int mpi_SendRecvAsync(const cudaqDistributedCommunicator_t *comm,
                             const void *sendbuf, void *recvbuf, int count,
                             DataType dataType, int peer, int32_t tag) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  if (PendingRequest::g_requests[comm].nActiveRequests != 0)
    return -1;
  MPI_Request *sendRequest = &(PendingRequest::g_requests[comm].requests[0]);
  MPI_Request *recvRequest = &(PendingRequest::g_requests[comm].requests[1]);
  int resSend = MPI_Isend(sendbuf, count, convertType(dataType), peer, tag,
                          unpackMpiCommunicator(comm), sendRequest);
  int resRecv = MPI_Irecv(recvbuf, count, convertType(dataType), peer, tag,
                          unpackMpiCommunicator(comm), recvRequest);
  if ((resSend != MPI_SUCCESS) || (resRecv != MPI_SUCCESS)) {
    MPI_Cancel(sendRequest);
    MPI_Cancel(recvRequest);
    return resSend != MPI_SUCCESS ? resSend : resRecv;
  }
  PendingRequest::g_requests[comm].nActiveRequests = 2;
  return 0;
}

/// @brief Wait for in-flight MPI_Isend and MPI_Irecv to complete
static int mpi_Synchronize(const cudaqDistributedCommunicator_t *comm) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  MPI_Status statuses[2];
  std::memset(statuses, 0, sizeof(statuses));
  int res = MPI_Waitall(PendingRequest::g_requests[comm].nActiveRequests,
                        PendingRequest::g_requests[comm].requests, statuses);
  PendingRequest::g_requests[comm].nActiveRequests = 0;
  return res;
}

/// @brief Wrapper of MPI_Abort
static int mpi_Abort(const cudaqDistributedCommunicator_t *comm,
                     int errorCode) {
  return MPI_Abort(unpackMpiCommunicator(comm), errorCode);
}

/// @brief Wrapper of MPI_Comm_dup
static int mpi_CommDup(const cudaqDistributedCommunicator_t *comm,
                       cudaqDistributedCommunicator_t **newDupComm) {
  // Use std::deque to make sure pointers to elements are valid.
  static std::deque<std::pair<MPI_Comm, cudaqDistributedCommunicator_t>>
      dup_comms;
  dup_comms.emplace_back(std::pair<MPI_Comm, cudaqDistributedCommunicator_t>());
  auto &[dupComm, newComm] = dup_comms.back();
  auto status = MPI_Comm_dup(unpackMpiCommunicator(comm), &dupComm);
  newComm.commPtr = &dupComm;
  newComm.commSize = sizeof(dupComm);
  *newDupComm = &newComm;
  return status;
}

/// @brief Wrapper of MPI_Comm_split
static int mpi_CommSplit(const cudaqDistributedCommunicator_t *comm,
                         int32_t color, int32_t key,
                         cudaqDistributedCommunicator_t **newSplitComm) {

  // Use std::deque to make sure pointers to elements are valid.
  static std::deque<std::pair<MPI_Comm, cudaqDistributedCommunicator_t>>
      split_comms;
  split_comms.emplace_back(
      std::pair<MPI_Comm, cudaqDistributedCommunicator_t>());
  auto &[splitComm, newComm] = split_comms.back();
  auto status =
      MPI_Comm_split(unpackMpiCommunicator(comm), color, key, &splitComm);
  newComm.commPtr = &splitComm;
  newComm.commSize = sizeof(splitComm);
  *newSplitComm = &newComm;
  return status;
}

/// @brief Return the underlying MPI_Comm as a type-erased object
cudaqDistributedCommunicator_t *getMpiCommunicator() {
  static MPI_Comm pluginComm = MPI_COMM_WORLD;
  static cudaqDistributedCommunicator_t commWorld{&pluginComm,
                                                  sizeof(pluginComm)};
  return &commWorld;
}

/// @brief Return the MPI shim interface (as a function table)
cudaqDistributedInterface_t *getDistributedInterface() {
  static cudaqDistributedInterface_t cudaqDistributedInterface{
      CUDAQ_DISTRIBUTED_INTERFACE_VERSION,
      mpi_initialize,
      mpi_finalize,
      mpi_initialized,
      mpi_finalized,
      mpi_getNumRanks,
      mpi_getProcRank,
      mpi_getCommSizeShared,
      mpi_Barrier,
      mpi_Bcast,
      mpi_Allreduce,
      mpi_AllreduceInplace,
      mpi_Allgather,
      mpi_AllgatherV,
      mpi_SendAsync,
      mpi_RecvAsync,
      mpi_SendRecvAsync,
      mpi_Synchronize,
      mpi_Abort,
      mpi_CommDup,
      mpi_CommSplit};
  return &cudaqDistributedInterface;
}
}
