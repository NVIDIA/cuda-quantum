/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "distributed_capi.h"
#include <string>
#include <vector>
namespace cudaq {
class MPIPlugin {
  cudaqDistributedInterface_t *m_distributedInterface;
  cudaqDistributedCommunicator_t *m_comm;
  bool m_valid;
public:
  static constexpr std::string_view COMM_GETTER_SYMBOL_NAME =
      "getMpiCommunicator";
  static constexpr std::string_view DISTRIBUTED_INTERFACE_GETTER_SYMBOL_NAME =
      "getDistributedInterface";
  MPIPlugin(const std::string &distributedInterfaceLib);
  cudaqDistributedInterface_t *get() { return m_distributedInterface; }

  void initialize();

  /// @brief Initialize MPI if available. This function
  /// is a no-op if there CUDA Quantum has not been built
  /// against MPI. Takes program arguments as input.
  void initialize(int argc, char **argv);

  /// @brief Return the rank of the calling process.
  int rank();

  /// @brief Return the number of MPI ranks.
  int num_ranks();

  /// @brief Return true if MPI is already initialized, false otherwise.
  bool is_initialized();

  /// @brief Return true if MPI is already finalized, false otherwise.
  bool is_finalized();

  /// @brief Gather all vector data locally into the provided
  /// global vector. Global vector must be sized to fit all
  /// vector elements coming from individual ranks.
  void all_gather(std::vector<double> &global,
                  const std::vector<double> &local);

  void broadcast(std::vector<double> &data, int rootRank);

  void all_reduce(std::vector<double> &global, const std::vector<double> &local, ReduceOp op);
  /// @brief Finalize MPI. This function
  /// is a no-op if there CUDA Quantum has not been built
  /// against MPI.
  void finalize();

  /// @brief Is the plugin valid?
  // Due to external runtime dependencies, e.g. Python modules, a loaded plugin
  // may not be valid and shouldn'd be used.
  bool isValid() const { return m_valid; }
};
} // namespace cudaq
