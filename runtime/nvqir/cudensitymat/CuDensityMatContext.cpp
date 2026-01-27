/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include "common/Logger.h"
#include "cudaq.h"
#include "cudaq/distributed/mpi_plugin.h"
#include <memory>
#include <mutex>

namespace cudaq::dynamics {
/// @brief Get the current CUDA context for the active device.
/// @return Context* Pointer to the current context.
Context *Context::getCurrentContext() {
  int currentDevice = -1;
  HANDLE_CUDA_ERROR(cudaGetDevice(&currentDevice));

  static std::unordered_map<int, std::unique_ptr<cudaq::dynamics::Context>>
      g_contexts;
  static std::mutex g_contextMutex;

  std::lock_guard<std::mutex> guard(g_contextMutex);
  const auto iter = g_contexts.find(currentDevice);
  if (iter == g_contexts.end()) {
    CUDAQ_INFO("Create cudensitymat context for device Id {}", currentDevice);
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
    if (m_scratchSpace) {
      cudaq::dynamics::DeviceAllocator::free(m_scratchSpace);
    }

    CUDAQ_INFO("Allocate scratch buffer of size {} bytes on device {}",
               minSizeBytes, m_deviceId);

    m_scratchSpace = cudaq::dynamics::DeviceAllocator::allocate(minSizeBytes);
    m_scratchSpaceSizeBytes = minSizeBytes;
  }

  return m_scratchSpace;
}

/// @brief Get the recommended workspace limit based on available memory.
/// @return std::size_t Recommended workspace limit in bytes.
std::size_t Context::getRecommendedWorkSpaceLimit() {
  std::size_t freeMem = 0;
  std::size_t totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  // Take 80% of free memory
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.80);
  return freeMem;
}

/// @brief Retrieve the MPI plugin comm interface
static cudaqDistributedInterface_t *getMpiPluginInterface() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  if (!mpiInterface)
    throw std::runtime_error("Invalid MPI distributed plugin encountered");
  return mpiInterface;
}

/// @brief Retrieve the MPI plugin (type-erased) comm pointer
static cudaqDistributedCommunicator_t *getMpiCommWrapper() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  if (!comm)
    throw std::runtime_error(
        "Invalid MPI distributed plugin communicator encountered");
  return comm;
}

/// @brief Construct a new Context object for a specific device.
/// @arg deviceId ID of the CUDA device.
Context::Context(int deviceId) : m_deviceId(deviceId) {
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  HANDLE_CUDM_ERROR(cudensitymatCreate(&m_cudmHandle));

  if (cudaq::mpi::is_initialized()) {
    cudaqDistributedInterface_t *mpiInterface = getMpiPluginInterface();
    cudaqDistributedCommunicator_t *comm = getMpiCommWrapper();
    cudaqDistributedCommunicator_t *dupComm = nullptr;
    const auto dupStatus = mpiInterface->CommDup(comm, &dupComm);
    if (dupStatus != 0 || dupComm == nullptr)
      throw std::runtime_error("Failed to duplicate the MPI communicator when "
                               "initializing cuDensityMat MPI");
    CUDAQ_INFO("cudensitymatResetDistributedConfiguration for handle {}\n",
               m_cudmHandle);
    HANDLE_CUDM_ERROR(cudensitymatResetDistributedConfiguration(
        m_cudmHandle, CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI, dupComm->commPtr,
        dupComm->commSize));
  }
  HANDLE_CUBLAS_ERROR(cublasCreate(&m_cublasHandle));
  m_opConverter = std::make_unique<CuDensityMatOpConverter>(m_cudmHandle);
}

bool Context::isDistributed() const { return getNumRanks() > 1; }

int Context::getNumRanks() const {
  return cudaq::mpi::is_initialized() ? cudaq::mpi::num_ranks() : 1;
}

int Context::getRank() const {
  return cudaq::mpi::is_initialized() ? cudaq::mpi::rank() : 0;
}

/// @brief Destroy the Context object and release resources.
Context::~Context() {
  m_opConverter.reset();
  cudensitymatDestroy(m_cudmHandle);
  cublasDestroy(m_cublasHandle);
  if (m_scratchSpaceSizeBytes > 0) {
    cudaq::dynamics::DeviceAllocator::free(m_scratchSpace);
  }
}
} // namespace cudaq::dynamics
