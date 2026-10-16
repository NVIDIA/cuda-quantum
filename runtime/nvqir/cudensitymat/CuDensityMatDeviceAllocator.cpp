/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatUtils.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optional>
#include <string>

namespace {

// The Driver API is required for the explicit fabric-qualified allocations
// used for zero-copy via UCX over MNNVL. Driver entry points are resolved
// through the CUDA runtime so that this library does not need to link against
// `libcuda` directly. Errors using the Driver API result in a warning message
// and a fallback to `cudaMalloc`.
struct DriverApi {
  decltype(&::cuDeviceGet) deviceGet = nullptr;
  decltype(&::cuDeviceGetAttribute) deviceGetAttribute = nullptr;
  decltype(&::cuMemGetAllocationGranularity) getAllocationGranularity = nullptr;
  decltype(&::cuMemCreate) create = nullptr;
  decltype(&::cuMemExportToShareableHandle) exportToShareableHandle = nullptr;
  decltype(&::cuMemAddressReserve) addressReserve = nullptr;
  decltype(&::cuMemAddressFree) addressFree = nullptr;
  decltype(&::cuMemMap) map = nullptr;
  decltype(&::cuMemUnmap) unmap = nullptr;
  decltype(&::cuMemSetAccess) setAccess = nullptr;
  decltype(&::cuMemRetainAllocationHandle) retainAllocationHandle = nullptr;
  decltype(&::cuMemRelease) release = nullptr;
  decltype(&::cuGetErrorName) getErrorName = nullptr;

  // Returns nullptr if the driver lacks the fabric memory APIs.
  static const DriverApi *get() {
    static const std::optional<DriverApi> api = load();
    return api ? &*api : nullptr;
  }

  std::string errorMessage(CUresult result) const {
    const char *name = nullptr;
    if (getErrorName(result, &name) != CUDA_SUCCESS || !name)
      name = "unknown";
    return "CUDA driver error " + std::string(name) + " (" +
           std::to_string(static_cast<int>(result)) + ")";
  }

private:
  static std::optional<DriverApi> load() {
    DriverApi api;
    if (resolve("cuGetErrorName", api.getErrorName) &&
        resolve("cuDeviceGet", api.deviceGet) &&
        resolve("cuDeviceGetAttribute", api.deviceGetAttribute) &&
        resolve("cuMemGetAllocationGranularity",
                api.getAllocationGranularity) &&
        resolve("cuMemCreate", api.create) &&
        resolve("cuMemExportToShareableHandle", api.exportToShareableHandle) &&
        resolve("cuMemAddressReserve", api.addressReserve) &&
        resolve("cuMemAddressFree", api.addressFree) &&
        resolve("cuMemMap", api.map) && resolve("cuMemUnmap", api.unmap) &&
        resolve("cuMemSetAccess", api.setAccess) &&
        resolve("cuMemRetainAllocationHandle", api.retainAllocationHandle) &&
        resolve("cuMemRelease", api.release))
      return api;
    return std::nullopt;
  }

  template <typename Fn>
  static bool resolve(const char *symbol, Fn &fn) {
    void *ptr = nullptr;
    cudaDriverEntryPointQueryResult status{};
    // Fabric memory handles require CUDA 12.3 or newer.
    if (cudaGetDriverEntryPointByVersion(symbol, &ptr, 12030, cudaEnableDefault,
                                         &status) != cudaSuccess ||
        status != cudaDriverEntryPointSuccess || !ptr)
      return false;
    fn = reinterpret_cast<Fn>(ptr);
    return true;
  }
};

// Fabric memory properties of the current device.
struct FabricDevice {
  int deviceId = -1;
  bool gpuDirectRdma = false;
  std::size_t granularity = 0;

  // Returns why the current device cannot allocate fabric memory, or nothing.
  std::optional<std::string> query(const DriverApi &api) {
    HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
    CUdevice cuDevice;
    int fabricSupported = 0;
    if (api.deviceGet(&cuDevice, deviceId) != CUDA_SUCCESS ||
        api.deviceGetAttribute(&fabricSupported,
                               CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                               cuDevice) != CUDA_SUCCESS ||
        !fabricSupported)
      return "device " + std::to_string(deviceId) +
             " does not support fabric memory handles";

    int rdmaSupported = 0;
    if (api.deviceGetAttribute(
            &rdmaSupported,
            CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
            cuDevice) == CUDA_SUCCESS)
      gpuDirectRdma = rdmaSupported != 0;

    const CUmemAllocationProp prop = allocationProp();
    const CUresult result = api.getAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS || granularity == 0)
      return "querying the fabric allocation granularity failed (" +
             api.errorMessage(result) + ")";
    return std::nullopt;
  }

  CUmemAllocationProp allocationProp() const {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    // Keeps the memory registrable for InfiniBand when MNNVL is unavailable.
    prop.allocFlags.gpuDirectRDMACapable = gpuDirectRdma ? 1 : 0;
    return prop;
  }

  std::size_t alignedSize(std::size_t sizeBytes) const {
    return (sizeBytes + granularity - 1) & ~(granularity - 1);
  }
};

CUresult createFabricMemory(const DriverApi &api, const FabricDevice &device,
                            std::size_t alignedSize, void **ptr) {
  const CUmemAllocationProp prop = device.allocationProp();
  CUmemGenericAllocationHandle handle = 0;
  CUresult result = api.create(&handle, alignedSize, &prop, 0);
  if (result != CUDA_SUCCESS)
    return result;

  // Allocation can succeed without a usable fabric (for example, without an
  // IMEX channel), so confirm that the handle is exportable.
  CUmemFabricHandle fabricHandle;
  result = api.exportToShareableHandle(&fabricHandle, handle,
                                       CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (result != CUDA_SUCCESS) {
    api.release(handle);
    return result;
  }

  CUdeviceptr devicePtr = 0;
  result =
      api.addressReserve(&devicePtr, alignedSize, device.granularity, 0, 0);
  if (result != CUDA_SUCCESS) {
    api.release(handle);
    return result;
  }

  result = api.map(devicePtr, alignedSize, 0, handle, 0);
  if (result != CUDA_SUCCESS) {
    api.addressFree(devicePtr, alignedSize);
    api.release(handle);
    return result;
  }
  // The mapping keeps the allocation alive until it is unmapped.
  api.release(handle);

  CUmemAccessDesc accessDesc = {};
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device.deviceId;
  result = api.setAccess(devicePtr, alignedSize, &accessDesc, 1);
  if (result != CUDA_SUCCESS) {
    api.unmap(devicePtr, alignedSize);
    api.addressFree(devicePtr, alignedSize);
    return result;
  }

  *ptr = reinterpret_cast<void *>(devicePtr);
  return CUDA_SUCCESS;
}

// Runs from destructors, including during process exit, so it must not throw.
void destroyFabricMemory(const DriverApi &api, void *ptr,
                         std::size_t alignedSize) noexcept {
  // Unmapping is not stream ordered.
  const cudaError_t syncResult = cudaDeviceSynchronize();
  if (syncResult != cudaSuccess && syncResult != cudaErrorCudartUnloading)
    CUDAQ_WARN("cudaDeviceSynchronize failed before unmapping fabric memory "
               "for a dynamics MPI buffer: {}",
               cudaGetErrorString(syncResult));
  const auto devicePtr = reinterpret_cast<CUdeviceptr>(ptr);
  api.unmap(devicePtr, alignedSize);
  api.addressFree(devicePtr, alignedSize);
}
} // namespace

std::optional<std::string>
cudaq::dynamics::DeviceAllocator::testFabricAllocation() {
  const DriverApi *api = DriverApi::get();
  if (!api)
    return "the CUDA driver lacks the fabric memory APIs of CUDA 12.3";
  FabricDevice device;
  if (auto unsupported = device.query(*api))
    return unsupported;
  void *ptr = nullptr;
  const CUresult result =
      createFabricMemory(*api, device, device.granularity, &ptr);
  if (result != CUDA_SUCCESS)
    return api->errorMessage(result);
  destroyFabricMemory(*api, ptr, device.granularity);
  return std::nullopt;
}

void *cudaq::dynamics::MpiBuffer::reserve(std::size_t sizeBytes,
                                          bool useFabricMemory) {
  if (sizeBytes <= m_sizeBytes)
    return m_data;
  reset();

  if (useFabricMemory) {
    PerfMetricScopeTimer metricTimer("MpiBuffer::reserve");
    const DriverApi *api = DriverApi::get();
    FabricDevice device;
    std::optional<std::string> failure =
        api ? device.query(*api)
            : "the CUDA driver lacks the fabric memory APIs of CUDA 12.3";
    if (!failure) {
      const std::size_t alignedSize = device.alignedSize(sizeBytes);
      const CUresult result =
          createFabricMemory(*api, device, alignedSize, &m_data);
      if (result == CUDA_SUCCESS) {
        m_sizeBytes = alignedSize;
        m_isFabricMemory = true;
        return m_data;
      }
      failure = api->errorMessage(result);
    }
    CUDAQ_WARN("Fabric memory allocation of {} bytes failed ({}); using "
               "cudaMalloc for this dynamics MPI buffer.",
               sizeBytes, *failure);
  }

  m_data = DeviceAllocator::allocate(sizeBytes);
  m_sizeBytes = sizeBytes;
  return m_data;
}

void cudaq::dynamics::MpiBuffer::release() {
  if (!m_isFabricMemory)
    reset();
}

void cudaq::dynamics::MpiBuffer::reset() {
  if (m_data) {
    if (m_isFabricMemory)
      destroyFabricMemory(*DriverApi::get(), m_data, m_sizeBytes);
    else
      DeviceAllocator::free(m_data);
  }
  m_data = nullptr;
  m_sizeBytes = 0;
  m_isFabricMemory = false;
}
