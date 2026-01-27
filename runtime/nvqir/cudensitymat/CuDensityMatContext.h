/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "CuDensityMatOpConverter.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudensitymat.h>

namespace cudaq::dynamics {
/// @brief Class representing the CUDAQ context for density matrix operations.
class Context {
public:
  Context(Context const &) = delete;
  Context &operator=(Context const &) = delete;
  ~Context();

  /// @brief Get the `cudensitymat` handle.
  /// @return cudensitymatHandle_t Handle to the `cudensitymat` context.
  cudensitymatHandle_t getHandle() const { return m_cudmHandle; }

  /// @brief Get the cuBLAS handle.
  /// @return cublasHandle_t Handle to the cuBLAS context.
  cublasHandle_t getCublasHandle() const { return m_cublasHandle; }

  /// @brief Get the operation converter.
  /// @return CuDensityMatOpConverter& Reference to the operation converter.
  CuDensityMatOpConverter &getOpConverter() { return *m_opConverter; }

  /// @brief Get the current CUDAQ context for the active device.
  /// @return Context* Pointer to the current context.
  static Context *getCurrentContext();

  /// @brief Get or allocate scratch space on the device.
  /// @arg minSizeBytes Minimum size of the scratch space in bytes.
  /// @return void* Pointer to the scratch space.
  void *getScratchSpace(std::size_t minSizeBytes);

  /// @brief Get the recommended workspace limit based on available memory.
  /// @return std::size_t Recommended workspace limit in bytes.
  static std::size_t getRecommendedWorkSpaceLimit();

  /// @brief Return the device Id of the context
  int getDeviceId() const { return m_deviceId; }

  /// @brief Return true if running in distributed mode
  bool isDistributed() const;

  /// @brief Return the total number of ranks in distributed mode
  // Returns 1 if not in distributed mode
  int getNumRanks() const;

  /// @brief Return the rank of this process in distributed mode
  /// Always returns 0 if not in distributed mode
  int getRank() const;

private:
  /// @brief Construct a new Context object for a specific device.
  /// @param deviceId ID of the CUDA device.
  Context(int deviceId);

  cudensitymatHandle_t m_cudmHandle;
  cublasHandle_t m_cublasHandle;
  std::unique_ptr<CuDensityMatOpConverter> m_opConverter;
  int m_deviceId;
  void *m_scratchSpace{nullptr};
  std::size_t m_scratchSpaceSizeBytes{0};
};
} // namespace cudaq::dynamics
