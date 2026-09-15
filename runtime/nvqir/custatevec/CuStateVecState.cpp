/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecState.h"

#include "CuStateVecCommunicator.h"
#include "CuStateVecError.h"

#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

/// Owns a temporary allocation on the active CUDA device.
struct DeviceBuffer {
  ~DeviceBuffer() {
    if (data)
      cudaFree(data);
  }
  void *data = nullptr;
};

/// Owns a temporary `cuStateVecEx` configuration dictionary.
class Dictionary {
public:
  ~Dictionary() {
    if (m_descriptor)
      custatevecExDictionaryDestroy(/*dictionary=*/m_descriptor);
  }
  custatevecExDictionaryDescriptor_t *address() { return &m_descriptor; }
  custatevecExDictionaryDescriptor_t get() const { return m_descriptor; }

private:
  custatevecExDictionaryDescriptor_t m_descriptor = nullptr;
};

/// Owns a non-blocking stream until ownership is transferred to a state.
class CudaStream {
public:
  CudaStream() {
    HANDLE_CUDA_ERROR(
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
  }
  ~CudaStream() {
    if (m_stream)
      cudaStreamDestroy(m_stream);
  }
  cudaStream_t get() const { return m_stream; }
  cudaStream_t release() { return std::exchange(m_stream, nullptr); }

private:
  cudaStream_t m_stream = nullptr;
};

} // namespace

namespace cudaq::cusv {

template <typename Scalar>
CuStateVecState<Scalar> CuStateVecState<Scalar>::createSingleDevice(
    int32_t maxWires, int32_t maxDeviceWires, int32_t deviceId,
    bool allowFp32Emulation) {
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  CudaStream stream;
  Dictionary configuration;
  HANDLE_CUSTATEVEC_ERROR(custatevecExConfigureStateVectorSingleDevice(
      /*svConfig=*/configuration.address(),
      /*svDataType=*/complexDataType<Scalar>(), /*numWires=*/maxWires,
      /*numDeviceWires=*/maxDeviceWires, /*deviceId=*/deviceId,
      /*capability=*/CUSTATEVEC_EX_SV_CAPABILITY_RESIZABLE));

  custatevecExStateVectorDescriptor_t state = nullptr;
  const cudaStream_t stateStream = stream.get();
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorCreateSingleProcess(
      /*stateVector=*/&state, /*svConfig=*/configuration.get(),
      /*streams=*/&stateStream, /*numStreams=*/1, /*resourceManager=*/nullptr));
  CuStateVecState result(state, stream.release());
  result.m_cloneFactory = [maxWires, maxDeviceWires, deviceId,
                           allowFp32Emulation] {
    return createSingleDevice(maxWires, maxDeviceWires, deviceId,
                              allowFp32Emulation);
  };
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorSetMathMode(
      /*stateVector=*/state,
      /*mode=*/allowFp32Emulation
          ? CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9
          : CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9));
  return result;
}
template <typename Scalar>
CuStateVecState<Scalar> CuStateVecState<Scalar>::createMultiProcess(
    int32_t maxWires, int32_t maxDeviceWires, int32_t deviceId,
    custatevecExMemorySharingMethod_t memorySharingMethod,
    const std::vector<custatevecExGlobalIndexBitClass_t> &globalIndexBitClasses,
    const std::vector<int32_t> &globalIndexBits,
    std::size_t transferWorkspaceSize,
    std::shared_ptr<CuStateVecCommunicator> communicator,
    bool allowFp32Emulation) {
  if (globalIndexBitClasses.size() != globalIndexBits.size())
    throw std::invalid_argument(
        "Each global-index-bit layer must have a communication class.");
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  CudaStream stream;
  Dictionary configuration;
  HANDLE_CUSTATEVEC_ERROR(custatevecExConfigureStateVectorMultiProcess(
      /*svConfig=*/configuration.address(),
      /*svDataType=*/complexDataType<Scalar>(), /*numWires=*/maxWires,
      /*numDeviceWires=*/maxDeviceWires, /*deviceId=*/deviceId,
      /*memorySharingMethod=*/memorySharingMethod,
      /*globalIndexBitClasses=*/globalIndexBitClasses.data(),
      /*numGlobalIndexBitsPerLayer=*/globalIndexBits.data(),
      /*numGlobalIndexBitLayers=*/static_cast<int32_t>(globalIndexBits.size()),
      /*transferWorkspaceSizeInBytes=*/transferWorkspaceSize,
      /*auxConfig=*/nullptr,
      /*capability=*/CUSTATEVEC_EX_SV_CAPABILITY_RESIZABLE));

  custatevecExStateVectorDescriptor_t state = nullptr;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorCreateMultiProcess(
      /*stateVector=*/&state, /*svConfig=*/configuration.get(),
      /*stream=*/stream.get(), /*exCommunicator=*/communicator->descriptor(),
      /*resourceManager=*/nullptr));
  CuStateVecState result(state, stream.release());
  result.m_communicator = communicator;
  result.m_cloneFactory = [maxWires, maxDeviceWires, deviceId,
                           memorySharingMethod, globalIndexBitClasses,
                           globalIndexBits, transferWorkspaceSize, communicator,
                           allowFp32Emulation] {
    return createMultiProcess(maxWires, maxDeviceWires, deviceId,
                              memorySharingMethod, globalIndexBitClasses,
                              globalIndexBits, transferWorkspaceSize,
                              communicator, allowFp32Emulation);
  };
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorSetMathMode(
      /*stateVector=*/state,
      /*mode=*/allowFp32Emulation
          ? CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9
          : CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9));
  return result;
}

template <typename Scalar>
CuStateVecState<Scalar>::~CuStateVecState() {
  reset();
}

template <typename Scalar>
CuStateVecState<Scalar>::CuStateVecState(CuStateVecState &&other) noexcept
    : m_communicator(std::move(other.m_communicator)),
      m_cloneFactory(std::move(other.m_cloneFactory)),
      m_state(std::exchange(other.m_state, nullptr)),
      m_stream(std::exchange(other.m_stream, nullptr)) {}

template <typename Scalar>
CuStateVecState<Scalar> &
CuStateVecState<Scalar>::operator=(CuStateVecState &&other) noexcept {
  if (this != &other) {
    reset();
    m_communicator = std::move(other.m_communicator);
    m_cloneFactory = std::move(other.m_cloneFactory);
    m_state = std::exchange(other.m_state, nullptr);
    m_stream = std::exchange(other.m_stream, nullptr);
  }
  return *this;
}

template <typename Scalar>
void CuStateVecState<Scalar>::reset() noexcept {
  if (m_state)
    custatevecExStateVectorDestroy(
        /*stateVector=*/std::exchange(m_state, nullptr));
  if (m_stream)
    cudaStreamDestroy(std::exchange(m_stream, nullptr));
}

template <typename Scalar>
custatevecExStateVectorDistributionType_t
CuStateVecState<Scalar>::distributionType() const {
  custatevecExStateVectorDistributionType_t value{};
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state,
      /*property=*/CUSTATEVEC_EX_SV_PROP_DISTRIBUTION_TYPE, /*value=*/&value,
      /*sizeInBytes=*/sizeof(value)));
  return value;
}

template <typename Scalar>
int32_t CuStateVecState<Scalar>::numWires() const {
  int32_t value = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state, /*property=*/CUSTATEVEC_EX_SV_PROP_NUM_WIRES,
      /*value=*/&value, /*sizeInBytes=*/sizeof(value)));
  return value;
}

template <typename Scalar>
int32_t CuStateVecState<Scalar>::numLocalWires() const {
  int32_t value = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state,
      /*property=*/CUSTATEVEC_EX_SV_PROP_NUM_LOCAL_WIRES, /*value=*/&value,
      /*sizeInBytes=*/sizeof(value)));
  return value;
}

template <typename Scalar>
int32_t CuStateVecState<Scalar>::numMigrationWires() const {
  int32_t value = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state,
      /*property=*/CUSTATEVEC_EX_SV_PROP_NUM_MIGRATION_WIRES, /*value=*/&value,
      /*sizeInBytes=*/sizeof(value)));
  return value;
}

template <typename Scalar>
int32_t CuStateVecState<Scalar>::maxLocalWires() const {
  int32_t value = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state,
      /*property=*/CUSTATEVEC_EX_SV_PROP_MAX_NUM_LOCAL_WIRES, /*value=*/&value,
      /*sizeInBytes=*/sizeof(value)));
  return value;
}

template <typename Scalar>
int32_t CuStateVecState<Scalar>::maxMigrationWires() const {
  int32_t value = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state,
      /*property=*/CUSTATEVEC_EX_SV_PROP_MAX_NUM_MIGRATION_WIRES,
      /*value=*/&value, /*sizeInBytes=*/sizeof(value)));
  return value;
}

template <typename Scalar>
std::vector<int32_t> CuStateVecState<Scalar>::wireOrdering() const {
  std::vector<int32_t> result(numWires());
  if (!result.empty())
    HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
        /*stateVector=*/m_state,
        /*property=*/CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
        /*value=*/result.data(),
        /*sizeInBytes=*/result.size() * sizeof(int32_t)));
  return result;
}

template <typename Scalar>
void CuStateVecState<Scalar>::exposeResources() const {
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorExposeResources(
      /*stateVector=*/m_state,
      /*exposeResources=*/CUSTATEVEC_EX_EXPOSE_RESOURCES_ACCESSIBLE));
}

template <typename Scalar>
std::vector<int32_t> CuStateVecState<Scalar>::subStateIndices() const {
  exposeResources();
  return queryInt32ArrayProperty(m_state, CUSTATEVEC_EX_SV_PROP_NUM_SUBSVS,
                                 CUSTATEVEC_EX_SV_PROP_SUBSV_INDICES);
}

template <typename Scalar>
std::vector<int32_t> CuStateVecState<Scalar>::deviceSubStateIndices() const {
  exposeResources();
  return queryInt32ArrayProperty(m_state,
                                 CUSTATEVEC_EX_SV_PROP_NUM_DEVICE_SUBSVS,
                                 CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES);
}

template <typename Scalar>
DeviceSubStateVector
CuStateVecState<Scalar>::deviceSubStateVector(int32_t index) const {
  DeviceSubStateVector result;
  HANDLE_CUSTATEVEC_ERROR(
      custatevecExStateVectorGetResourcesFromDeviceSubSVView(
          /*stateVector=*/m_state, /*subSVIndex=*/index,
          /*deviceId=*/&result.deviceId, /*d_subSV=*/&result.data,
          /*stream=*/&result.stream, /*handle=*/&result.handle));
  return result;
}

template <typename Scalar>
DeviceSubStateVector
CuStateVecState<Scalar>::writableDeviceSubStateVector(int32_t index) {
  DeviceSubStateVector result;
  void *data = nullptr;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetResourcesFromDeviceSubSV(
      /*stateVector=*/m_state, /*subSVIndex=*/index,
      /*deviceId=*/&result.deviceId, /*d_subSV=*/&data,
      /*stream=*/&result.stream, /*handle=*/&result.handle));
  result.data = data;
  return result;
}

template <typename Scalar>
CuStateVecState<Scalar> CuStateVecState<Scalar>::cloneEmpty() const {
  if (!m_cloneFactory)
    throw std::runtime_error("State layout cannot be cloned.");
  auto result = m_cloneFactory();
  result.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, numLocalWires());
  result.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION,
                  numMigrationWires());
  const int32_t globalWires =
      numWires() - numLocalWires() - numMigrationWires();
  result.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE, globalWires);
  return result;
}

template <typename Scalar>
void CuStateVecState<Scalar>::copyFrom(const CuStateVecState &other) {
  if (this == &other)
    return;
  if (distributionType() != other.distributionType() ||
      numWires() != other.numWires() ||
      numLocalWires() != other.numLocalWires() ||
      numMigrationWires() != other.numMigrationWires())
    throw std::invalid_argument("Cannot copy incompatible Ex state layouts.");
  other.synchronize();
  // `other` may carry a different wire ordering than this state. This is the
  // norm on the noisy-trajectory reinit path: the previous trajectory leaves
  // the working state's ordering drifted from the saved snapshot we copy back
  // from here. Relabel this state's ordering to match the source (metadata-only
  // reassign, no amplitude movement) so the per-slice copy below lines up
  // amplitudes position-by-position.
  const auto sourceOrdering = other.wireOrdering();
  if (wireOrdering() != sourceOrdering)
    reassignWireOrdering(sourceOrdering);

  // Exposing each descriptor flushes deferred migration work and captures the
  // staged/unstaged layout used to select the host-resident copies.
  const auto indices = subStateIndices();
  if (indices != other.subStateIndices())
    throw std::invalid_argument(
        "Cannot copy states with different sub-state assignments.");
  const auto destinationUnstaged = queryInt32ArrayProperty(
      descriptor(), CUSTATEVEC_EX_SV_PROP_NUM_UNSTAGED_SUBSVS,
      CUSTATEVEC_EX_SV_PROP_UNSTAGED_SUBSV_INDICES);
  const auto sourceUnstaged = queryInt32ArrayProperty(
      other.descriptor(), CUSTATEVEC_EX_SV_PROP_NUM_UNSTAGED_SUBSVS,
      CUSTATEVEC_EX_SV_PROP_UNSTAGED_SUBSV_INDICES);

  const std::size_t bytes =
      (std::size_t{1} << numLocalWires()) * sizeof(std::complex<Scalar>);
  int32_t numSlices = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/m_state,
      /*property=*/CUSTATEVEC_EX_SV_PROP_NUM_SUBSV_SLICES, /*value=*/&numSlices,
      /*sizeInBytes=*/sizeof(numSlices)));
  if (numSlices < 1 || bytes % static_cast<std::size_t>(numSlices) != 0)
    throw std::runtime_error("Invalid Ex sub-state slice layout.");
  const std::size_t sliceBytes = bytes / numSlices;

  // Preserve each descriptor's current placement. Host-to-device and
  // device-to-host copies use the resident side directly instead of staging
  // the other side and causing a second migration.
  // Each sub-state is copied according to where both sides currently reside,
  // covering the four host/device placement combinations without restaging.
  for (const int32_t index : indices) {
    const bool sourceOnHost = contains(sourceUnstaged, index);
    const bool destinationOnHost = contains(destinationUnstaged, index);
    // Both sides unstaged on host: copy each host slice directly.
    if (sourceOnHost && destinationOnHost) {
      for (int32_t slice = 0; slice < numSlices; ++slice) {
        const void *source = nullptr;
        void *destination = nullptr;
        custatevecExMemoryPlacement_t sourcePlacement{};
        custatevecExMemoryPlacement_t destinationPlacement{};
        int32_t sourceDevice = -1;
        int32_t destinationDevice = -1;
        cudaStream_t sourceStream = nullptr;
        cudaStream_t destinationStream = nullptr;
        HANDLE_CUSTATEVEC_ERROR(
            custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(
                /*stateVector=*/other.descriptor(), /*subSVIndex=*/index,
                /*sliceIndex=*/slice, /*subSVSlice=*/&source,
                /*placement=*/&sourcePlacement, /*deviceId=*/&sourceDevice,
                /*stream=*/&sourceStream));
        HANDLE_CUSTATEVEC_ERROR(
            custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(
                /*stateVector=*/m_state, /*subSVIndex=*/index,
                /*sliceIndex=*/slice, /*subSVSlice=*/&destination,
                /*placement=*/&destinationPlacement,
                /*deviceId=*/&destinationDevice,
                /*stream=*/&destinationStream));
        if (sourcePlacement != CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST ||
            destinationPlacement != CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST)
          throw std::runtime_error(
              "Expected unstaged Ex sub-state slices in host memory.");
        std::memcpy(destination, source, sliceBytes);
      }
      continue;
    }

    // Destination sub-state is device-resident.
    if (!destinationOnHost) {
      const auto destination = writableDeviceSubStateVector(index);
      HANDLE_CUDA_ERROR(cudaSetDevice(destination.deviceId));
      // Source also on device: one contiguous peer copy (same device required).
      if (!sourceOnHost) {
        const auto source = other.deviceSubStateVector(index);
        if (source.deviceId != destination.deviceId)
          throw std::invalid_argument(
              "Cannot copy Ex states placed on different devices.");
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(
            const_cast<void *>(destination.data), source.data, bytes,
            cudaMemcpyDeviceToDevice, destination.stream));
        continue;
      }
      // Source on host: stream each host slice into the device sub-state.
      for (int32_t slice = 0; slice < numSlices; ++slice) {
        const void *source = nullptr;
        custatevecExMemoryPlacement_t sourcePlacement{};
        int32_t sourceDevice = -1;
        cudaStream_t sourceStream = nullptr;
        HANDLE_CUSTATEVEC_ERROR(
            custatevecExStateVectorGetResourcesFromUnstagedSubSVSliceView(
                /*stateVector=*/other.descriptor(), /*subSVIndex=*/index,
                /*sliceIndex=*/slice, /*subSVSlice=*/&source,
                /*placement=*/&sourcePlacement, /*deviceId=*/&sourceDevice,
                /*stream=*/&sourceStream));
        if (sourcePlacement != CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST ||
            sourceDevice != destination.deviceId)
          throw std::runtime_error("Invalid host-to-device state placement.");
        auto *const destinationSlice =
            static_cast<char *>(const_cast<void *>(destination.data)) +
            static_cast<std::size_t>(slice) * sliceBytes;
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(destinationSlice, source, sliceBytes,
                                          cudaMemcpyHostToDevice,
                                          destination.stream));
      }
      continue;
    }

    // Destination on host, source on device: stream each device slice to host.
    const auto source = other.deviceSubStateVector(index);
    for (int32_t slice = 0; slice < numSlices; ++slice) {
      void *destination = nullptr;
      custatevecExMemoryPlacement_t destinationPlacement{};
      int32_t destinationDevice = -1;
      cudaStream_t destinationStream = nullptr;
      HANDLE_CUSTATEVEC_ERROR(
          custatevecExStateVectorGetResourcesFromUnstagedSubSVSlice(
              /*stateVector=*/m_state, /*subSVIndex=*/index,
              /*sliceIndex=*/slice, /*subSVSlice=*/&destination,
              /*placement=*/&destinationPlacement,
              /*deviceId=*/&destinationDevice,
              /*stream=*/&destinationStream));
      if (destinationPlacement != CUSTATEVEC_EX_MEMORY_PLACEMENT_ON_HOST ||
          destinationDevice != source.deviceId)
        throw std::runtime_error("Invalid device-to-host state placement.");
      HANDLE_CUDA_ERROR(cudaSetDevice(destinationDevice));
      const auto *const sourceSlice =
          static_cast<const char *>(source.data) +
          static_cast<std::size_t>(slice) * sliceBytes;
      HANDLE_CUDA_ERROR(cudaMemcpyAsync(destination, sourceSlice, sliceBytes,
                                        cudaMemcpyDeviceToHost,
                                        destinationStream));
    }
  }
  synchronize();
}

template <typename Scalar>
void CuStateVecState<Scalar>::stageSubStateVector(int32_t index) const {
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorStageSubSV(
      /*stateVector=*/m_state, /*subSVIndex=*/index));
}

template <typename Scalar>
void CuStateVecState<Scalar>::addWires(custatevecExIndexBitDomain_t domain,
                                       int32_t count) {
  if (count == 0)
    return;
  std::vector<int32_t> addedWires(count);
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorAddWires(
      /*stateVector=*/m_state, /*indexBitDomain=*/domain,
      /*numWiresToAdd=*/count,
      /*wireInitMode=*/CUSTATEVEC_EX_WIRE_INIT_MODE_ZERO,
      /*wiresAdded=*/addedWires.data()));
}

template <typename Scalar>
void CuStateVecState<Scalar>::setZeroState() {
  HANDLE_CUSTATEVEC_ERROR(
      custatevecExStateVectorSetZeroState(/*stateVector=*/m_state));
}

template <typename Scalar>
void CuStateVecState<Scalar>::synchronize() const {
  HANDLE_CUSTATEVEC_ERROR(
      custatevecExStateVectorSynchronize(/*stateVector=*/m_state));
}

template <typename Scalar>
void CuStateVecState<Scalar>::getState(std::complex<Scalar> *data,
                                       custatevecIndex_t begin,
                                       custatevecIndex_t end,
                                       int32_t maxConcurrentCopies) const {
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetState(
      /*stateVector=*/m_state, /*state=*/data,
      /*dataType=*/complexDataType<Scalar>(), /*begin=*/begin, /*end=*/end,
      /*maxNumConcurrentCopies=*/maxConcurrentCopies));
}

template <typename Scalar>
void CuStateVecState<Scalar>::setState(const std::complex<Scalar> *data,
                                       custatevecIndex_t begin,
                                       custatevecIndex_t end,
                                       int32_t maxConcurrentCopies) {
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorSetState(
      /*stateVector=*/m_state, /*state=*/data,
      /*dataType=*/complexDataType<Scalar>(), /*begin=*/begin, /*end=*/end,
      /*maxNumConcurrentCopies=*/maxConcurrentCopies));
}

template <typename Scalar>
bool CuStateVecState<Scalar>::setStateFromDevicePointer(const void *data,
                                                        std::size_t size) {
  if (!data)
    throw std::invalid_argument("State-vector data pointer cannot be null.");
  cudaPointerAttributes attributes{};
  const cudaError_t status =
      cudaPointerGetAttributes(&attributes, const_cast<void *>(data));
  if (status != cudaSuccess || (attributes.type != cudaMemoryTypeDevice &&
                                attributes.type != cudaMemoryTypeManaged)) {
    if (status != cudaSuccess)
      cudaGetLastError();
    return false;
  }
  if (size != (std::size_t{1} << numWires()))
    return false;

  const std::size_t subStateSize = std::size_t{1} << numLocalWires();
  const std::size_t bytes = subStateSize * sizeof(std::complex<Scalar>);
  const auto *const source = static_cast<const std::complex<Scalar> *>(data);
  for (const int32_t index : subStateIndices()) {
    stageSubStateVector(index);
    const auto destination = writableDeviceSubStateVector(index);
    const auto *const sourceSubState =
        source + static_cast<std::size_t>(index) * subStateSize;
    HANDLE_CUDA_ERROR(cudaSetDevice(destination.deviceId));
    if (attributes.type == cudaMemoryTypeDevice &&
        attributes.device != destination.deviceId) {
      HANDLE_CUDA_ERROR(cudaMemcpyPeerAsync(
          const_cast<void *>(destination.data), destination.deviceId,
          sourceSubState, attributes.device, bytes, destination.stream));
    } else {
      HANDLE_CUDA_ERROR(cudaMemcpyAsync(const_cast<void *>(destination.data),
                                        sourceSubState, bytes,
                                        cudaMemcpyDefault, destination.stream));
    }
  }
  synchronize();
  return true;
}

template <typename Scalar>
bool CuStateVecState<Scalar>::appendState(const void *data, std::size_t size) {
  if (!data || size == 0 || !std::has_single_bit(size))
    throw std::invalid_argument("Invalid state to append.");
  const int32_t addedWires = static_cast<int32_t>(std::countr_zero(size));
  if (distributionType() != CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE ||
      numMigrationWires() != 0 ||
      numLocalWires() + addedWires > maxLocalWires())
    return false;

  const auto indices = deviceSubStateIndices();
  if (indices.size() != 1)
    return false;
  const int32_t index = indices.front();
  stageSubStateVector(index);
  const auto current = deviceSubStateVector(index);
  HANDLE_CUDA_ERROR(cudaSetDevice(current.deviceId));
  const std::size_t oldSize = std::size_t{1} << numLocalWires();
  const std::size_t oldBytes = oldSize * sizeof(std::complex<Scalar>);
  const std::size_t addedBytes = size * sizeof(std::complex<Scalar>);

  DeviceBuffer oldState;
  DeviceBuffer copiedAddedState;
  HANDLE_CUDA_ERROR(cudaMalloc(&oldState.data, oldBytes));
  HANDLE_CUDA_ERROR(cudaMemcpyAsync(oldState.data, current.data, oldBytes,
                                    cudaMemcpyDeviceToDevice, current.stream));

  cudaPointerAttributes attributes{};
  const cudaError_t pointerStatus =
      cudaPointerGetAttributes(&attributes, const_cast<void *>(data));
  const bool directlyAccessible = pointerStatus == cudaSuccess &&
                                  (attributes.type == cudaMemoryTypeManaged ||
                                   (attributes.type == cudaMemoryTypeDevice &&
                                    attributes.device == current.deviceId));
  const auto *addedState = static_cast<const std::complex<Scalar> *>(data);
  if (!directlyAccessible) {
    if (pointerStatus != cudaSuccess)
      cudaGetLastError();
    HANDLE_CUDA_ERROR(cudaMalloc(&copiedAddedState.data, addedBytes));
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(copiedAddedState.data, data, addedBytes,
                                      cudaMemcpyDefault, current.stream));
    addedState =
        static_cast<const std::complex<Scalar> *>(copiedAddedState.data);
  }
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(current.stream));

  addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, addedWires);
  stageSubStateVector(index);
  const auto destination = writableDeviceSubStateVector(index);
  if (oldSize > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) ||
      size > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("State extension exceeds cuBLAS range.");
  const std::size_t resultBytes = oldSize * size * sizeof(std::complex<Scalar>);
  HANDLE_CUDA_ERROR(cudaMemsetAsync(const_cast<void *>(destination.data), 0,
                                    resultBytes, destination.stream));
  CublasHandle handle;
  // Bind cuBLAS to the sub-state's stream so the rank-1 update below runs on
  // the same stream as the surrounding device copies (ordered, no cross-stream
  // sync).
  HANDLE_CUBLAS_ERROR(cublasSetStream(/*handle=*/handle.get(),
                                      /*streamId=*/destination.stream));
  // Append a state via the tensor product `oldState (x) addedState`, computed
  // as a rank-1 outer-product update A = alpha * x * y^T into the destination
  // buffer (Cgeru_64 for single precision, Zgeru_64 for double precision).
  if constexpr (std::is_same_v<Scalar, float>) {
    const cuFloatComplex alpha = make_cuFloatComplex(1.0F, 0.0F);
    HANDLE_CUBLAS_ERROR(cublasCgeru_64(
        /*handle=*/handle.get(), /*m=*/static_cast<int64_t>(oldSize),
        /*n=*/static_cast<int64_t>(size), /*alpha=*/&alpha,
        /*x=*/static_cast<const cuFloatComplex *>(oldState.data), /*incx=*/1,
        /*y=*/reinterpret_cast<const cuFloatComplex *>(addedState), /*incy=*/1,
        /*A=*/
        static_cast<cuFloatComplex *>(const_cast<void *>(destination.data)),
        /*lda=*/static_cast<int64_t>(oldSize)));
  } else {
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    HANDLE_CUBLAS_ERROR(cublasZgeru_64(
        /*handle=*/handle.get(), /*m=*/static_cast<int64_t>(oldSize),
        /*n=*/static_cast<int64_t>(size), /*alpha=*/&alpha,
        /*x=*/static_cast<const cuDoubleComplex *>(oldState.data), /*incx=*/1,
        /*y=*/reinterpret_cast<const cuDoubleComplex *>(addedState), /*incy=*/1,
        /*A=*/
        static_cast<cuDoubleComplex *>(const_cast<void *>(destination.data)),
        /*lda=*/static_cast<int64_t>(oldSize)));
  }
  synchronize();
  return true;
}

template <typename Scalar>
void CuStateVecState<Scalar>::permuteIndexBits(
    const std::vector<int32_t> &permutation) {
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorPermuteIndexBits(
      /*stateVector=*/m_state, /*permutation=*/permutation.data(),
      /*permutationLen=*/static_cast<int32_t>(permutation.size()),
      /*permutationType=*/CUSTATEVEC_EX_PERMUTATION_SCATTER));
}

template <typename Scalar>
void CuStateVecState<Scalar>::normalizeWireOrdering() {
  const auto ordering = wireOrdering();
  for (std::size_t index = 0; index < ordering.size(); ++index) {
    if (ordering[index] == static_cast<int32_t>(index))
      continue;
    permuteIndexBits(ordering);
    synchronize();
    return;
  }
}

template <typename Scalar>
void CuStateVecState<Scalar>::reassignWireOrdering(
    const std::vector<int32_t> &ordering) {
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorReassignWireOrdering(
      /*stateVector=*/m_state, /*wireOrdering=*/ordering.data(),
      /*wireOrderingLen=*/static_cast<int32_t>(ordering.size())));
}

template class CuStateVecState<float>;
template class CuStateVecState<double>;

} // namespace cudaq::cusv
