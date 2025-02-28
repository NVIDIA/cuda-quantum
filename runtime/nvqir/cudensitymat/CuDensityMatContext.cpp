/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "common/Logger.h"
#include <memory>
#include <mutex>

namespace {
static std::unordered_map<int, std::unique_ptr<cudaq::dynamics::Context>>
    g_contexts;
static std::mutex g_contextMutex;
} // namespace

namespace cudaq {
namespace dynamics {

/// @brief Get the current CUDA context for the active device.
/// @return Context* Pointer to the current context.
Context *Context::getCurrentContext() {
  int currentDevice = -1;
  HANDLE_CUDA_ERROR(cudaGetDevice(&currentDevice));
  std::lock_guard<std::mutex> guard(g_contextMutex);
  const auto iter = g_contexts.find(currentDevice);
  if (iter == g_contexts.end()) {
    cudaq::info("Create cudensitymat context for device Id {}", currentDevice);
    const auto [insertedIter, success] = g_contexts.emplace(std::make_pair(
        currentDevice, std::unique_ptr<Context>(new Context(currentDevice))));
    if (!success)
      throw std::runtime_error("Failed to create cudensitymat context");
    return insertedIter->second.get();
  }

  return iter->second.get();
}

/// @brief Get or allocate scratch space on the device.
/// @arg minSizeBytes Minimum size of the scratch space in bytes.
/// @return void* Pointer to the scratch space.
void *Context::getScratchSpace(std::size_t minSizeBytes) {
  if (minSizeBytes > m_scratchSpaceSizeBytes) {
    // Realloc
    if (m_scratchSpace)
      HANDLE_CUDA_ERROR(cudaFree(m_scratchSpace));

    cudaq::info("Allocate scratch buffer of size {} bytes on device {}",
                minSizeBytes, m_deviceId);

    HANDLE_CUDA_ERROR(cudaMalloc(&m_scratchSpace, minSizeBytes));
    m_scratchSpaceSizeBytes = minSizeBytes;
  }

  return m_scratchSpace;
}

/// @brief Get the recommended workspace limit based on available memory.
/// @return std::size_t Recommended workspace limit in bytes.
std::size_t Context::getRecommendedWorkSpaceLimit() {
  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  // Take 80% of free memory
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.80);
  return freeMem;
}

/// @brief Construct a new Context object for a specific device.
/// @arg deviceId ID of the CUDA device.
Context::Context(int deviceId) : m_deviceId(deviceId) {
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  HANDLE_CUDM_ERROR(cudensitymatCreate(&m_cudmHandle));
  HANDLE_CUBLAS_ERROR(cublasCreate(&m_cublasHandle));
  m_opConverter = std::make_unique<CuDensityMatOpConverter>(m_cudmHandle);
}

/// @brief Destroy the Context object and release resources.
Context::~Context() {
  m_opConverter.reset();
  cudensitymatDestroy(m_cudmHandle);
  cublasDestroy(m_cublasHandle);
  if (m_scratchSpaceSizeBytes > 0)
    cudaFree(m_scratchSpace);
}
} // namespace dynamics
} // namespace cudaq
