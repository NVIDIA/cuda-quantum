/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <cstddef>
#include <stdint.h>

/*! \file distributed_capi.h
    \brief CUDA-Q Shim C-API for MPI support

    This header file defines a wrapper interface for MPI functionalities that
   CUDA-Q and its backends need. The interface is defined in a
   MPI-independent manner so that CUDA-Q libraries doesn't need to be
   linked against a particular MPI implementation. MPI support will be provided
   at runtime via dynamical library loading.
*/

extern "C" {
/// @brief  Type-erasure representation of a MPI communicator (MPI_Comm)
typedef struct {
  /// @brief  Pointer to the encapsulated MPI_Comm
  /// i.e., MPI_Comm* -> void * conversion.
  void *commPtr;
  /// @brief Size of MPI_Comm type for checking/verification purposes.
  std::size_t commSize;
} cudaqDistributedCommunicator_t;

#define CUDAQ_DISTRIBUTED_INTERFACE_VERSION 1

/// @brief Data type that we support
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

/// @brief Type of MPI reduce ops that we support
// Plugin implementation need to convert it to MPI enum as needed.
enum ReduceOp { SUM, PROD, MIN, MIN_LOC };

/// @brief Encapsulates MPI functionalities as a function table.
// Plugin implementations will redirect these functions into proper MPI API
// calls.
typedef struct {
  /// @brief  Version number for compatibility checking.
  int version;
  /// @brief  MPI_Init
  int (*initialize)(int32_t *, char ***);
  /// @brief  MPI_Finalize
  int (*finalize)();
  /// @brief MPI_Initialized
  int (*initialized)(int32_t *);
  /// @brief MPI_Finalized
  int (*finalized)(int32_t *);
  /// @brief MPI_Comm_size
  int (*getNumRanks)(const cudaqDistributedCommunicator_t *, int32_t *);
  /// @brief MPI_Comm_rank
  int (*getProcRank)(const cudaqDistributedCommunicator_t *, int32_t *);
  /// Returns the size of the local subgroup of processes sharing node memory
  int (*getCommSizeShared)(const cudaqDistributedCommunicator_t *comm,
                           int32_t *);
  /// @brief MPI_Barrier
  int (*Barrier)(const cudaqDistributedCommunicator_t *);
  /// @brief MPI_Bcast
  int (*Bcast)(const cudaqDistributedCommunicator_t *, void *, int32_t,
               DataType, int32_t);
  /// @brief MPI_Allreduce
  int (*Allreduce)(const cudaqDistributedCommunicator_t *, const void *, void *,
                   int32_t, DataType, ReduceOp);
  /// @brief MPI_Allreduce with MPI_IN_PLACE
  int (*AllreduceInPlace)(const cudaqDistributedCommunicator_t *, void *,
                          int32_t, DataType, ReduceOp);
  /// @brief MPI_Allgather
  int (*Allgather)(const cudaqDistributedCommunicator_t *, const void *, void *,
                   int32_t, DataType);
  /// @brief MPI_Allgatherv
  int (*AllgatherV)(const cudaqDistributedCommunicator_t *, const void *,
                    int32_t, void *, const int32_t *, const int32_t *,
                    DataType);
  /// @brief MPI_Isend
  /// @note The MPI plugin API allows for a maximum of two concurrent
  /// non-blocking requests (e.g., one `Isend` and one `Irecv`). Hence,
  /// `Synchronize` should be called as appropriate to resolve in-flight
  /// requests before making new ones.
  int (*SendAsync)(const cudaqDistributedCommunicator_t *, const void *, int,
                   DataType, int, int32_t);
  /// @brief MPI_Irecv
  /// @note The MPI plugin API allows for a maximum of two concurrent
  /// non-blocking requests (e.g., one `Isend` and one `Irecv`). Hence,
  /// `Synchronize` should be called as appropriate to resolve in-flight
  /// requests before making new ones.
  int (*RecvAsync)(const cudaqDistributedCommunicator_t *, void *, int,
                   DataType, int, int32_t);
  /// @brief MPI_Isend and MPI_Irecv in one call
  /// @note The MPI plugin API allows for a maximum of two concurrent
  /// non-blocking requests (e.g., one `Isend` and one `Irecv`). Since this
  /// `SendRecvAsync` creates two pending requests, `Synchronize` must be called
  /// to resolve in-flight requests before making new ones.
  int (*SendRecvAsync)(const cudaqDistributedCommunicator_t *, const void *,
                       void *, int, DataType, int, int32_t);
  /// @brief Wait for previous non-blocking MPI_Isend and MPI_Irecv to complete
  int (*Synchronize)(const cudaqDistributedCommunicator_t *);
  /// @brief MPI_Abort
  int (*Abort)(const cudaqDistributedCommunicator_t *, int);
  /// @brief MPI_Comm_dup
  int (*CommDup)(const cudaqDistributedCommunicator_t *,
                 cudaqDistributedCommunicator_t **);
  /// @brief MPI_Comm_split
  int (*CommSplit)(const cudaqDistributedCommunicator_t *, int32_t, int32_t,
                   cudaqDistributedCommunicator_t **);
} cudaqDistributedInterface_t;
}