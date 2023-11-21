/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "distributed_capi.h"
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mpi.h>
namespace {
bool initCalledByThis = false;
MPI_Datatype convertType(DataType dataType) {
  switch (dataType) {
  case FLOAT_32:
    return MPI_FLOAT;
  case FLOAT_64:
    return MPI_DOUBLE;
  }
  __builtin_unreachable();
}

MPI_Op convertType(ReduceOp opType) {
  switch (opType) {
  case SUM:
    return MPI_SUM;
  case PROD:
    return MPI_PROD;
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
} // namespace
extern "C" {

int mpi_initialize(int32_t *argc, char ***argv) {
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

int mpi_finalize() {
  if (!initCalledByThis)
    return MPI_SUCCESS;
  return MPI_Finalize();
}

int mpi_initialized(int32_t *flag) { return MPI_Initialized(flag); }
int mpi_finalized(int32_t *flag) { return MPI_Finalized(flag); }

int mpi_getNumRanks(const cudaqDistributedCommunicator_t *comm, int32_t *size) {
  return MPI_Comm_size(unpackMpiCommunicator(comm), size);
}

int mpi_getProcRank(const cudaqDistributedCommunicator_t *comm, int32_t *rank) {
  return MPI_Comm_rank(unpackMpiCommunicator(comm), rank);
}
int mpi_Barrier(const cudaqDistributedCommunicator_t *comm) {
  return MPI_Barrier(unpackMpiCommunicator(comm));
}
int mpi_Bcast(const cudaqDistributedCommunicator_t *comm, void *buffer,
              int32_t count, DataType dataType, int32_t rootRank) {
  return MPI_Bcast(buffer, count, convertType(dataType), rootRank,
                   unpackMpiCommunicator(comm));
}
int mpi_Allreduce(const cudaqDistributedCommunicator_t *comm,
                  const void *sendBuffer, void *recvBuffer, int32_t count,
                  DataType dataType, ReduceOp opType) {
  return MPI_Allreduce(sendBuffer, recvBuffer, count, convertType(dataType),
                       convertType(opType), unpackMpiCommunicator(comm));
}
int mpi_Allgather(const cudaqDistributedCommunicator_t *comm,
                  const void *sendBuffer, void *recvBuffer, int32_t count,
                  DataType dataType) {
  return MPI_Allgather(sendBuffer, count, convertType(dataType), recvBuffer,
                       count, convertType(dataType),
                       unpackMpiCommunicator(comm));
}
int mpi_CommDup(const cudaqDistributedCommunicator_t *comm,
                cudaqDistributedCommunicator_t **newDupComm) {
  MPI_Comm dupComm;
  auto status = MPI_Comm_dup(unpackMpiCommunicator(comm), &dupComm);
  // Use std::deque to make sure pointers to elements are valid.
  static std::deque<cudaqDistributedCommunicator_t> dup_comms;
  dup_comms.emplace_back(cudaqDistributedCommunicator_t());
  auto &newComm = dup_comms.back();
  newComm.commPtr = &dupComm;
  newComm.commSize = sizeof(dupComm);
  *newDupComm = &newComm;
  return status;
}

int mpi_CommSplit(const cudaqDistributedCommunicator_t *comm, int32_t color,
                  int32_t key, cudaqDistributedCommunicator_t **newSplitComm) {
  MPI_Comm splitComm;
  auto status =
      MPI_Comm_split(unpackMpiCommunicator(comm), color, key, &splitComm);
  // Use std::deque to make sure pointers to elements are valid.
  static std::deque<cudaqDistributedCommunicator_t> split_comms;
  split_comms.emplace_back(cudaqDistributedCommunicator_t());
  auto &newComm = split_comms.back();
  newComm.commPtr = &splitComm;
  newComm.commSize = sizeof(splitComm);
  *newSplitComm = &newComm;
  return status;
}

cudaqDistributedCommunicator_t *getMpiCommunicator() {
  static MPI_Comm pluginComm = MPI_COMM_WORLD;
  static cudaqDistributedCommunicator_t commWorld{&pluginComm,
                                                  sizeof(pluginComm)};
  return &commWorld;
}

cudaqDistributedInterface_t *getDistributedInterface() {
  static cudaqDistributedInterface_t cudaqDistributedInterface{
      CUDAQ_DISTRIBUTED_INTERFACE_VERSION,
      mpi_initialize,
      mpi_finalize,
      mpi_initialized,
      mpi_finalized,
      mpi_getNumRanks,
      mpi_getProcRank,
      mpi_Barrier,
      mpi_Bcast,
      mpi_Allreduce,
      mpi_Allgather,
      mpi_CommDup,
      mpi_CommSplit};
  return &cudaqDistributedInterface;
}
}
