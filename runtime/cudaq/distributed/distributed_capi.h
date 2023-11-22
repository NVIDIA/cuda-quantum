/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <cstddef>
#include <stdint.h>
extern "C" {
typedef struct {
  void *commPtr;
  std::size_t commSize;
} cudaqDistributedCommunicator_t;

#define CUDAQ_DISTRIBUTED_INTERFACE_VERSION 1

// Data type that we need to support.
// Plugin implementation need to convert it to MPI data type enum as needed.
enum DataType {
  INT_8,
  INT_16,
  INT_32,
  INT_64,
  FLOAT_32,
  FLOAT_64,
  FLOAT_COMPLEX,
  DOUBLE_COMPLEX
};

// Type of reduce ops
enum ReduceOp { SUM, PROD, MIN, MIN_LOC };

typedef struct {
  int version;
  int (*initialize)(int32_t *, char ***);
  int (*finalize)();
  int (*initialized)(int32_t *);
  int (*finalized)(int32_t *);
  int (*getNumRanks)(const cudaqDistributedCommunicator_t *, int32_t *);
  int (*getProcRank)(const cudaqDistributedCommunicator_t *, int32_t *);
  // Returns the size of the local subgroup of processes sharing node memory
  int (*getCommSizeShared)(const cudaqDistributedCommunicator_t *comm,
                           int32_t *);
  int (*Barrier)(const cudaqDistributedCommunicator_t *);
  int (*Bcast)(const cudaqDistributedCommunicator_t *, void *, int32_t,
               DataType, int32_t);
  int (*Allreduce)(const cudaqDistributedCommunicator_t *, const void *, void *,
                   int32_t, DataType, ReduceOp);
  int (*AllreduceInPlace)(const cudaqDistributedCommunicator_t *, void *,
                          int32_t, DataType, ReduceOp);
  int (*Allgather)(const cudaqDistributedCommunicator_t *, const void *, void *,
                   int32_t, DataType);
  int (*AllgatherV)(const cudaqDistributedCommunicator_t *, const void *,
                    int32_t, void *, const int32_t *, const int32_t *,
                    DataType);
  int (*SendAsync)(const cudaqDistributedCommunicator_t *, const void *, int,
                   DataType, int, int32_t);
  int (*RecvAsync)(const cudaqDistributedCommunicator_t *, void *, int,
                   DataType, int, int32_t);
  int (*SendRecvAsync)(const cudaqDistributedCommunicator_t *, const void *,
                       void *, int, DataType, int, int32_t);
  int (*Synchronize)(const cudaqDistributedCommunicator_t *);
  int (*Abort)(const cudaqDistributedCommunicator_t *, int);
  int (*CommDup)(const cudaqDistributedCommunicator_t *,
                 cudaqDistributedCommunicator_t **);
  int (*CommSplit)(const cudaqDistributedCommunicator_t *, int32_t, int32_t,
                   cudaqDistributedCommunicator_t **);
} cudaqDistributedInterface_t;
}