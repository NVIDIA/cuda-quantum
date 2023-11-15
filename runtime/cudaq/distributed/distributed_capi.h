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
enum DataType { FLOAT_32, FLOAT_64 };

// Type of reduce ops
enum ReduceOp { SUM, PROD };

typedef struct {
  int version;
  int (*initialize)(int32_t *, char ***);
  int (*finalize)();
  int (*initialized)(int32_t *);
  int (*finalized)(int32_t *);
  int (*getNumRanks)(const cudaqDistributedCommunicator_t *, int32_t *);
  int (*getProcRank)(const cudaqDistributedCommunicator_t *, int32_t *);
  int (*Barrier)(const cudaqDistributedCommunicator_t *);
  int (*Bcast)(const cudaqDistributedCommunicator_t *, void *, int32_t,
               DataType, int32_t);
  int (*Allreduce)(const cudaqDistributedCommunicator_t *, const void *, void *,
                   int32_t, DataType, ReduceOp);
  int (*Allgather)(const cudaqDistributedCommunicator_t *, const void *, void *,
                   int32_t, DataType);
  int (*CommDup)(const cudaqDistributedCommunicator_t *,
                 cudaqDistributedCommunicator_t **);
  int (*CommSplit)(const cudaqDistributedCommunicator_t *, int32_t, int32_t,
                   cudaqDistributedCommunicator_t **);
} cudaqDistributedInterface_t;

// cudaqDistributedCommunicator_t *getMpiCommunicator();
// cudaqDistributedInterface_t *getDistributedInterface();
}