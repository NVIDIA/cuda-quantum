/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
  std::string m_libFile;

public:
  static constexpr std::string_view COMM_GETTER_SYMBOL_NAME =
      "getMpiCommunicator";
  static constexpr std::string_view DISTRIBUTED_INTERFACE_GETTER_SYMBOL_NAME =
      "getDistributedInterface";
  // Static method to safely check whether a path contains an usable MPI
  // inteface library.
  static bool isValidInterfaceLib(const std::string &distributedInterfaceLib);
  MPIPlugin(const std::string &distributedInterfaceLib);
  cudaqDistributedInterface_t *get() { return m_distributedInterface; }
  cudaqDistributedCommunicator_t *getComm() { return m_comm; }
  std::string getPluginPath() const { return m_libFile; }
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

  /// @brief Return true if MPI is already finalized, false otherwise.
  bool is_finalized();

  /// @brief Gather all vector data (floating point numbers) locally into the
  /// provided global vector.
  ///
  /// Global vector must be sized to fit all
  /// vector elements coming from individual ranks.
  void all_gather(std::vector<double> &global,
                  const std::vector<double> &local);

  /// @brief Gather all vector data (integers) locally into the provided
  /// global vector.
  ///
  /// Global vector must be sized to fit all
  /// vector elements coming from individual ranks.
  void all_gather(std::vector<int> &global, const std::vector<int> &local);

  /// @brief Broadcast a vector from a root rank to all other ranks
  void broadcast(std::vector<double> &data, int rootRank);

  /// @brief Broadcast a string from a root rank to all other ranks
  void broadcast(std::string &data, int rootRank);

  /// @brief Combines local vector data from all processes and distributes the
  /// result back to all processes into the provided global vector.
  void all_reduce(std::vector<double> &global, const std::vector<double> &local,
                  ReduceOp op);

  /// @brief Finalize MPI. This function
  /// is a no-op if there CUDA-Q has not been built
  /// against MPI.
  void finalize();

  /// @brief Is the plugin valid?
  // Due to external runtime dependencies, e.g. Python modules, a loaded plugin
  // may not be valid and shouldn'd be used.
  bool isValid() const { return m_valid; }
};

namespace mpi {
/// @brief Retrieve the runtime MPI plugin.
/// @note Throw an error if no runtime MPI plugin is available unless `unsafe`
/// is true.
/// @param unsafe If true, returns a `nullptr` rather than throwing an error if
/// no MPI plugin is available. Hence, the caller needs to check the returned
/// pointer before use.
/// @return Pointer to the runtime MPI plugin
extern ::cudaq::MPIPlugin *getMpiPlugin(bool unsafe = false);
} // namespace mpi
} // namespace cudaq
