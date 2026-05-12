/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq {
namespace mpi {
/// @brief Return true if CUDA-Q has MPI plugin support.
bool available();

/// @brief Initialize MPI if available. This function
/// is a no-op if there CUDA-Q has not been built
/// against MPI.
void initialize();

/// @brief Initialize MPI if available. This function
/// is a no-op if there CUDA-Q has not been built
/// against MPI. Takes program arguments as input.
void initialize(int argc, char **argv);

/// @brief Return the rank of the calling process.
int rank();

/// @brief Return the number of MPI ranks.
int num_ranks();

/// @brief Return true if MPI is already initialized, false otherwise.
bool is_initialized();

namespace details {
#define CUDAQ_ALL_REDUCE_DEF(TYPE, BINARY)                                     \
  TYPE allReduce(const TYPE &, const BINARY<TYPE> &);

CUDAQ_ALL_REDUCE_DEF(float, std::plus)
CUDAQ_ALL_REDUCE_DEF(float, std::multiplies)

CUDAQ_ALL_REDUCE_DEF(double, std::plus)
CUDAQ_ALL_REDUCE_DEF(double, std::multiplies)

} // namespace details

/// @brief Reduce all values across ranks with the specified binary function.
template <typename T, typename BinaryFunction>
T all_reduce(const T &localValue, const BinaryFunction &function) {
  return details::allReduce(localValue, function);
}

/// @brief Gather all vector data (floating point numbers) locally into the
/// provided global vector.
///
/// Global vector must be sized to fit all vector
/// elements coming from individual ranks.
void all_gather(std::vector<double> &global, const std::vector<double> &local);

/// @brief Gather all vector data (integers) locally into the provided
/// global vector.
///
/// Global vector must be sized to fit all
/// vector elements coming from individual ranks.
void all_gather(std::vector<int> &global, const std::vector<int> &local);

/// @brief Broadcast a vector from a process (rootRank) to all other processes.
void broadcast(std::vector<double> &data, int rootRank);

/// @brief Broadcast a string from a process (rootRank) to all other processes.
void broadcast(std::string &data, int rootRank);

/// @brief Duplicate the communicator. Returns the new communicator (as a void*)
/// and its size.
std::pair<void *, std::size_t> comm_dup();

/// @brief Finalize MPI. This function
/// is a no-op if there CUDA-Q has not been built
/// against MPI.
void finalize();

} // namespace mpi
} // namespace cudaq
