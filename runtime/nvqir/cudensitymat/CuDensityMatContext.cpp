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
#include "cudaq/cudaq_mpi.h"
#include "cudaq/distributed/mpi_plugin.h"
#include "cudaq/runtime/logger/logger.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

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
  if (minSizeBytes > m_scratchSpace.sizeBytes())
    CUDAQ_INFO("Allocate scratch buffer of size {} bytes on device {}",
               minSizeBytes, m_deviceId);
  return m_scratchSpace.reserve(minSizeBytes, m_useFabricMemory);
}

void *Context::getExpectationResultBuffer(std::size_t minSizeBytes) {
  return m_expectationResult.reserve(minSizeBytes, m_useFabricMemory);
}

void Context::releaseExpectationResultBuffer() {
  m_expectationResult.release();
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

int32_t detail::gpuFabricDomainSize(std::string fabric, int32_t numRanks,
                                    int32_t ranksPerNode) {
  std::transform(fabric.begin(), fabric.end(), fabric.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (fabric == "MNNVL")
    return numRanks;
  if (fabric == "NVL")
    return ranksPerNode;
  if (fabric == "NONE")
    return 1;
  int32_t domainSize = 0;
  const char *const begin = fabric.data();
  const char *const end = begin + fabric.size();
  const auto [position, error] = std::from_chars(begin, end, domainSize);
  if (error != std::errc{} || position != end || domainSize < 1)
    throw std::invalid_argument(
        "CUDAQ_GPU_FABRIC must be MNNVL, NVL, NONE, or a positive integer "
        "domain size.");
  return domainSize;
}

bool detail::requestsFabricMemory(const char *fabric, int32_t numRanks,
                                  int32_t ranksPerNode) {
  if (!fabric)
    return false;
  const int32_t domainSize =
      gpuFabricDomainSize(fabric, numRanks, ranksPerNode);
  return ranksPerNode < numRanks && domainSize >= numRanks;
}

/// @brief Decide whether MPI buffers use fabric-exportable memory, which allows
/// zero-copy via UCX. This requires more than one node and a `CUDAQ_GPU_FABRIC`
/// NVLink domain that spans every rank. If a test allocation fails on any rank,
/// all ranks fall back to `cudaMalloc` and the lowest failing rank prints a
/// warning.
static bool useFabricMemory(cudaqDistributedInterface_t *mpiInterface,
                            const cudaqDistributedCommunicator_t *comm) {
  const char *fabric = std::getenv("CUDAQ_GPU_FABRIC");
  if (!fabric)
    return false;
  int32_t numRanks = 0;
  int32_t rank = 0;
  int32_t ranksPerNode = 0;
  if (mpiInterface->getNumRanks(comm, &numRanks) != 0 ||
      mpiInterface->getProcRank(comm, &rank) != 0 ||
      mpiInterface->getCommSizeShared(comm, &ranksPerNode) != 0)
    throw std::runtime_error("Failed to query the MPI communicator topology");
  if (!detail::requestsFabricMemory(fabric, numRanks, ranksPerNode))
    return false;

  const auto failure = DeviceAllocator::testFabricAllocation();
  int32_t firstFailedRank = failure ? rank : numRanks;
  if (mpiInterface->AllreduceInPlace(comm, &firstFailedRank, 1, INT_32, MIN) !=
      0)
    throw std::runtime_error("Failed to reduce the fabric memory test result");
  if (firstFailedRank == numRanks)
    return true;
  if (rank == firstFailedRank)
    CUDAQ_WARN("CUDAQ_GPU_FABRIC={} requests fabric memory for dynamics MPI "
               "buffers, but a test allocation failed on rank {}: {}. Using "
               "cudaMalloc on all ranks instead.",
               fabric, rank, *failure);
  return false;
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
    m_useFabricMemory = useFabricMemory(mpiInterface, dupComm);
    CUDAQ_INFO("Fabric memory for dynamics MPI buffers is {}.",
               m_useFabricMemory ? "enabled" : "disabled");
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

bool Context::setMpiCommunicator(void *comm, int commSizeBytes) {
  cudaqDistributedInterface_t *mpiInterface = getMpiPluginInterface();
  return cudensitymatResetDistributedConfiguration(
             m_cudmHandle, CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI, comm,
             commSizeBytes) == CUDENSITYMAT_STATUS_SUCCESS;
}

/// @brief Destroy the Context object and release resources.
Context::~Context() {
  m_opConverter.reset();
  cudensitymatDestroy(m_cudmHandle);
  cublasDestroy(m_cublasHandle);
  m_scratchSpace.reset();
  m_expectationResult.reset();
}
} // namespace cudaq::dynamics
