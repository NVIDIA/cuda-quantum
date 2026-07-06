/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecError.h"

#include <custatevecEx.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace cudaq::cusv {

class CuStateVecCommunicator;

/// Device resources backing one currently staged Ex sub-state vector.
struct DeviceSubStateVector {
  int32_t deviceId = 0;
  const void *data = nullptr;
  cudaStream_t stream = nullptr;
  custatevecHandle_t handle = nullptr;
};

template <typename Scalar>
constexpr cudaDataType_t complexDataType() {
  static_assert(std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, double>);
  if constexpr (std::is_same_v<Scalar, float>)
    return CUDA_C_32F;
  return CUDA_C_64F;
}

/// True if `values` contains `value`.
template <typename T>
bool contains(const std::vector<T> &values, const T &value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

/// Query an int32 array property of a distributed descriptor: read the element
/// count via `countProp`, then fill the array via `arrayProp`.
inline std::vector<int32_t>
queryInt32ArrayProperty(custatevecExStateVectorDescriptor_t state,
                        custatevecExStateVectorProperty_t countProp,
                        custatevecExStateVectorProperty_t arrayProp) {
  int32_t count = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
      /*stateVector=*/state, /*property=*/countProp, /*value=*/&count,
      /*sizeInBytes=*/sizeof(count)));
  std::vector<int32_t> result(count);
  if (count != 0)
    HANDLE_CUSTATEVEC_ERROR(custatevecExStateVectorGetProperty(
        /*stateVector=*/state, /*property=*/arrayProp, /*value=*/result.data(),
        /*sizeInBytes=*/result.size() * sizeof(int32_t)));
  return result;
}

/// @brief Move-only owner of a `cuStateVecEx` state-vector descriptor.
///
/// This class represents the simulator's internal live state and exposes the
/// low-level operations needed by the gate engines and the single- and
/// multi-process simulators, including resizing, host-device migration, wire
/// ordering, synchronization, and state-vector transfers. It intentionally
/// does not implement `cudaq::SimulationState`. When CUDA-Q needs to return a
/// user-visible state, ownership of this object is moved into a
/// `CuStateVecSimulationState` adapter.
template <typename Scalar>
class CuStateVecState {
public:
  static CuStateVecState createSingleDevice(int32_t maxWires,
                                            int32_t maxDeviceWires,
                                            int32_t deviceId,
                                            bool allowFp32Emulation);
  static CuStateVecState
  createMultiProcess(int32_t maxWires, int32_t maxDeviceWires, int32_t deviceId,
                     custatevecExMemorySharingMethod_t memorySharingMethod,
                     const std::vector<custatevecExGlobalIndexBitClass_t>
                         &globalIndexBitClasses,
                     const std::vector<int32_t> &globalIndexBits,
                     std::size_t transferWorkspaceSize,
                     std::shared_ptr<CuStateVecCommunicator> communicator,
                     bool allowFp32Emulation);

  ~CuStateVecState();
  CuStateVecState(CuStateVecState &&other) noexcept;
  CuStateVecState &operator=(CuStateVecState &&other) noexcept;
  CuStateVecState(const CuStateVecState &) = delete;
  CuStateVecState &operator=(const CuStateVecState &) = delete;

  custatevecExStateVectorDescriptor_t descriptor() const { return m_state; }
  int32_t numWires() const;
  custatevecExStateVectorDistributionType_t distributionType() const;
  int32_t numLocalWires() const;
  int32_t numMigrationWires() const;
  int32_t maxLocalWires() const;
  int32_t maxMigrationWires() const;
  std::vector<int32_t> wireOrdering() const;
  std::vector<int32_t> subStateIndices() const;
  std::vector<int32_t> deviceSubStateIndices() const;
  DeviceSubStateVector deviceSubStateVector(int32_t index) const;
  DeviceSubStateVector writableDeviceSubStateVector(int32_t index);
  CuStateVecState cloneEmpty() const;
  void copyFrom(const CuStateVecState &other);
  std::shared_ptr<CuStateVecCommunicator> communicator() const {
    return m_communicator;
  }

  void addWires(custatevecExIndexBitDomain_t domain, int32_t count);
  void setZeroState();
  void exposeResources() const;
  void stageSubStateVector(int32_t index) const;
  void synchronize() const;
  void getState(std::complex<Scalar> *data, custatevecIndex_t begin,
                custatevecIndex_t end, int32_t maxConcurrentCopies = 1) const;
  void setState(const std::complex<Scalar> *data, custatevecIndex_t begin,
                custatevecIndex_t end, int32_t maxConcurrentCopies = 1);
  /// Imports a complete state directly from CUDA-accessible memory. For a
  /// distributed state, each rank copies only its assigned global sub-states.
  bool setStateFromDevicePointer(const void *data, std::size_t size);

  /// Appends an arbitrary state with a device-resident Kronecker product when
  /// the resized state remains entirely local. Returns false when resizing
  /// requires host migration or distribution, so the caller can use the
  /// general matrix-based path.
  bool appendState(const void *data, std::size_t size);
  void permuteIndexBits(const std::vector<int32_t> &permutation);

  /// If the wire ordering is not the identity, permute the index bits back to
  /// it and synchronize; otherwise a no-op.
  void normalizeWireOrdering();
  void reassignWireOrdering(const std::vector<int32_t> &ordering);

private:
  explicit CuStateVecState(custatevecExStateVectorDescriptor_t state,
                           cudaStream_t stream)
      : m_state(state), m_stream(stream) {}
  void reset() noexcept;

  std::shared_ptr<CuStateVecCommunicator> m_communicator;
  std::function<CuStateVecState()> m_cloneFactory;
  custatevecExStateVectorDescriptor_t m_state = nullptr;
  cudaStream_t m_stream = nullptr;
};

extern template class CuStateVecState<float>;
extern template class CuStateVecState<double>;

} // namespace cudaq::cusv
