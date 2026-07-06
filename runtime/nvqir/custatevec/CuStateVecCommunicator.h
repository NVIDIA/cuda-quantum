/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecConfig.h"

#include <custatevecEx.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace cudaq::cusv {

/// Resolved communicator implementation used by `cuStateVecEx`.
enum class CommunicatorProvider { Cudaq, OpenMPI, MPICH, External };

/// @brief Owns a `cuStateVecEx` communicator and its provider lifetime.
///
/// The wrapper initializes the selected built-in or CUDA-Q external provider,
/// creates the communicator descriptor, and permits the CUDA-Q communicator to
/// be selected when the external provider is used.
class CuStateVecCommunicator {
public:
  CuStateVecCommunicator(CommunicatorPlugin plugin,
                         const std::string &mpiLibrary);
  ~CuStateVecCommunicator();
  CuStateVecCommunicator(const CuStateVecCommunicator &) = delete;
  CuStateVecCommunicator &operator=(const CuStateVecCommunicator &) = delete;

  custatevecExCommunicatorDescriptor_t descriptor() const;
  int32_t size() const;
  int32_t rank() const;
  CommunicatorProvider provider() const { return m_provider; }
  void broadcast(void *buffer, int32_t count, cudaDataType_t dataType,
                 int32_t root) const;
  void allReduce(const void *sendBuffer, void *receiveBuffer, int32_t count,
                 cudaDataType_t dataType) const;
  void allGather(const void *sendBuffer, void *receiveBuffer, int32_t count,
                 cudaDataType_t dataType) const;
  void setCommunicator(void *communicator, std::size_t communicatorSize);
  void reset() noexcept;

  /// Finalize the process-wide communicator provider (MPI-level).
  static void finalizeProvider() noexcept;

private:
  custatevecExCommunicatorDescriptor_t communicator() const;
  void initializeProvider(const std::string &mpiLibrary);
  void create();
  void destroy() noexcept;

  custatevecExCommunicatorDescriptor_t m_communicator = nullptr;
  bool m_active = false;
  CommunicatorProvider m_provider = CommunicatorProvider::Cudaq;
};

} // namespace cudaq::cusv
