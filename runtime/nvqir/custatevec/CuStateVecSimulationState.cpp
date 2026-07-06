/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecSimulationState.h"

#include "CuStateVecCommunicator.h"
#include "CuStateVecError.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace {

// Validate that a state-vector length is a non-zero power of two and return its
// qubit count (log2).
std::size_t checkedQubitCount(std::size_t size) {
  if (size == 0 || (size & (size - 1)) != 0)
    throw std::invalid_argument(
        "State-vector data size must be a non-zero power of two.");
  return std::countr_zero(size);
}

// Conjugate inner product <left|right> of two same-device sub-state vectors,
// computed on the GPU with cuBLAS (Cdotc/Zdotc).
template <typename Scalar>
std::complex<double> deviceInnerProduct(
    cublasHandle_t handle, const cudaq::cusv::DeviceSubStateVector &left,
    const cudaq::cusv::DeviceSubStateVector &right, std::size_t elementCount) {
  if (left.deviceId != right.deviceId)
    throw std::invalid_argument(
        "Cannot overlap sub-state vectors on different devices.");
  if (elementCount >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("Sub-state vector exceeds cuBLAS range.");

  HANDLE_CUDA_ERROR(cudaSetDevice(left.deviceId));
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(right.stream));
  HANDLE_CUBLAS_ERROR(
      cublasSetStream(/*handle=*/handle, /*streamId=*/left.stream));
  if constexpr (std::is_same_v<Scalar, float>) {
    cuFloatComplex result{};
    HANDLE_CUBLAS_ERROR(cublasCdotc_64(
        /*handle=*/handle, /*n=*/static_cast<int64_t>(elementCount),
        /*x=*/static_cast<const cuFloatComplex *>(left.data), /*incx=*/1,
        /*y=*/static_cast<const cuFloatComplex *>(right.data), /*incy=*/1,
        /*result=*/&result));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(left.stream));
    return {static_cast<double>(cuCrealf(result)),
            static_cast<double>(cuCimagf(result))};
  } else {
    cuDoubleComplex result{};
    HANDLE_CUBLAS_ERROR(cublasZdotc_64(
        /*handle=*/handle, /*n=*/static_cast<int64_t>(elementCount),
        /*x=*/static_cast<const cuDoubleComplex *>(left.data), /*incx=*/1,
        /*y=*/static_cast<const cuDoubleComplex *>(right.data), /*incy=*/1,
        /*result=*/&result));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(left.stream));
    return {cuCreal(result), cuCimag(result)};
  }
}

// Narrow an element count to the int32 a collective (all-gather/all-reduce)
// expects, checking for overflow.
int32_t checkedCollectiveCount(std::size_t count) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
    throw std::overflow_error("Collective element count exceeds int32 range.");
  return static_cast<int32_t>(count);
}

} // namespace

namespace cudaq::cusv {

template <typename Scalar>
std::size_t CuStateVecSimulationState<Scalar>::numElements() const {
  return getNumElements();
}

template <typename Scalar>
std::size_t CuStateVecSimulationState<Scalar>::getNumQubits() const {
  if (!m_state)
    return 0;
  return m_state->numWires();
}

// Locate one amplitude's physical slot without normalizing the whole state:
// logical wire `ordering[b]` fills physical bit `b` (identity ordering =>
// no-op). e.g. an ordering swapping two wires maps logical 0b01 to physical
// 0b10.
template <typename Scalar>
std::size_t CuStateVecSimulationState<Scalar>::physicalIndex(
    std::size_t logicalIndex) const {
  std::size_t result = 0;
  const auto ordering = state().wireOrdering();
  for (std::size_t indexBit = 0; indexBit < ordering.size(); ++indexBit) {
    const auto logicalWire = static_cast<std::size_t>(ordering[indexBit]);
    result |= ((logicalIndex >> logicalWire) & std::size_t{1}) << indexBit;
  }
  return result;
}

template <typename Scalar>
void CuStateVecSimulationState<Scalar>::normalizeWireOrdering() const {
  if (m_scalarDevicePtr)
    return;
  m_state->normalizeWireOrdering();
}

// Gather this rank's sub-state vectors (each 2^numLocalWires amplitudes) into
// one contiguous host vector, ordered by the sub-state indices this rank holds.
template <typename Scalar>
std::vector<std::complex<Scalar>>
CuStateVecSimulationState<Scalar>::localState() const {
  const std::size_t subStateSize = std::size_t{1} << state().numLocalWires();
  const auto indices = state().subStateIndices();
  std::vector<std::complex<Scalar>> result(subStateSize * indices.size());
  for (std::size_t position = 0; position < indices.size(); ++position) {
    const std::size_t begin =
        static_cast<std::size_t>(indices[position]) * subStateSize;
    state().getState(result.data() + position * subStateSize, begin,
                     begin + subStateSize);
  }
  state().synchronize();
  return result;
}

template <typename Scalar>
template <typename HostScalar>
void CuStateVecSimulationState<Scalar>::copyToHost(
    std::complex<HostScalar> *data, std::size_t numElementsRequested) const {
  static_assert(std::is_same_v<HostScalar, Scalar>);
  if (!data)
    throw std::invalid_argument("Destination state pointer cannot be null.");
  if (m_scalarDevicePtr) {
    if (numElementsRequested != 1)
      throw std::invalid_argument("Invalid destination state size.");
    HANDLE_CUDA_ERROR(cudaMemcpy(data, m_scalarDevicePtr, sizeof(*data),
                                 cudaMemcpyDeviceToHost));
    return;
  }
  normalizeWireOrdering();

  if (state().distributionType() !=
      CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS) {
    if (numElementsRequested != numElements())
      throw std::invalid_argument("Invalid destination state size.");
    state().getState(data, 0, numElementsRequested);
    state().synchronize();
    return;
  }

  const auto indices = state().subStateIndices();
  const auto local = localState();
  // A rank-local buffer requests a per-rank view; a global-sized buffer
  // requests collection and ordering of all distributed sub-state vectors.
  if (numElementsRequested == local.size()) {
    std::copy(local.begin(), local.end(), data);
    return;
  }
  if (numElementsRequested != numElements())
    throw std::invalid_argument(
        "Distributed destination must have rank-local or global size.");

  const auto communicator = state().communicator();
  if (!communicator)
    throw std::runtime_error("Distributed state has no communicator.");
  const std::size_t rankCount = static_cast<std::size_t>(communicator->size());
  std::vector<int32_t> gatheredIndices(indices.size() * rankCount);
  communicator->allGather(indices.data(), gatheredIndices.data(),
                          checkedCollectiveCount(indices.size()), CUDA_R_32I);
  std::vector<std::complex<Scalar>> rankOrdered(local.size() * rankCount);
  communicator->allGather(local.data(), rankOrdered.data(),
                          checkedCollectiveCount(local.size()),
                          complexDataType<Scalar>());

  const std::size_t subStateSize = std::size_t{1} << state().numLocalWires();
  if (gatheredIndices.size() * subStateSize != numElementsRequested)
    throw std::runtime_error("Distributed state layout is inconsistent.");
  // Scatter the rank-ordered gathered amplitudes into global sub-state order.
  std::vector<bool> assigned(gatheredIndices.size(), false);
  for (std::size_t position = 0; position < gatheredIndices.size();
       ++position) {
    const int32_t index = gatheredIndices[position];
    if (index < 0 || static_cast<std::size_t>(index) >= assigned.size() ||
        assigned[index])
      throw std::runtime_error("Invalid distributed sub-state assignment.");
    assigned[index] = true;
    std::copy_n(rankOrdered.data() + position * subStateSize, subStateSize,
                data + static_cast<std::size_t>(index) * subStateSize);
  }
}

template <typename Scalar>
std::complex<double>
CuStateVecSimulationState<Scalar>::overlap(const SimulationState &other) {
  const auto *const otherState =
      dynamic_cast<const CuStateVecSimulationState<Scalar> *>(&other);
  if (!otherState)
    throw std::invalid_argument(
        "Overlap requires another cuStateVec state of the same precision.");
  if (m_scalarDevicePtr || otherState->m_scalarDevicePtr)
    throw std::invalid_argument(
        "Cannot overlap a zero-qubit cuStateVec state.");
  if (other.getNumQubits() != getNumQubits())
    throw std::invalid_argument("Cannot overlap states with different sizes.");

  // Amplitudes are compared position-by-position, so both operands must share
  // the same logical wire ordering and the same storage layout.
  normalizeWireOrdering();
  otherState->normalizeWireOrdering();
  if (state().distributionType() != otherState->state().distributionType() ||
      state().numLocalWires() != otherState->state().numLocalWires() ||
      state().numMigrationWires() != otherState->state().numMigrationWires())
    throw std::invalid_argument(
        "Cannot overlap states with different storage layouts.");

  const auto indices = state().subStateIndices();
  if (indices != otherState->state().subStateIndices())
    throw std::invalid_argument(
        "Cannot overlap states with different sub-state assignments.");
  const std::size_t localElements = std::size_t{1} << state().numLocalWires();

  // Accumulate dot product of conj(this) and other over each sub-state vector.
  // Migrated sub-SVs are staged onto the device (rotating them through the
  // device buffer) so every pair reduces to a single cuBLAS conjugate dot on
  // the GPU.
  std::complex<double> localResult{};
  std::optional<CublasHandle> deviceHandle;
  for (const int32_t index : indices) {
    state().stageSubStateVector(index);
    otherState->state().stageSubStateVector(index);
    const auto left = state().deviceSubStateVector(index);
    const auto right = otherState->state().deviceSubStateVector(index);
    if (!deviceHandle) {
      HANDLE_CUDA_ERROR(cudaSetDevice(left.deviceId));
      deviceHandle.emplace();
    }
    localResult += deviceInnerProduct<Scalar>(deviceHandle->get(), left, right,
                                              localElements);
  }

  // Each rank holds a partial sum over its own sub-states; reduce across ranks
  // and return the overlap magnitude |<this|other>|.
  if (state().distributionType() ==
      CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS) {
    const auto communicator = state().communicator();
    if (!communicator)
      throw std::runtime_error("Distributed state has no communicator.");
    std::complex<double> globalResult{};
    communicator->allReduce(&localResult, &globalResult, 1, CUDA_C_64F);
    return std::abs(globalResult);
  }
  return std::abs(localResult);
}

template <typename Scalar>
std::complex<double>
CuStateVecSimulationState<Scalar>::amplitudeAt(std::size_t logicalIndex) {
  if (m_scalarDevicePtr) {
    if (logicalIndex != 0)
      throw std::out_of_range("Invalid zero-qubit state element index.");
    std::complex<Scalar> result;
    HANDLE_CUDA_ERROR(cudaMemcpy(&result, m_scalarDevicePtr, sizeof(result),
                                 cudaMemcpyDeviceToHost));
    return {result.real(), result.imag()};
  }
  const std::size_t index = physicalIndex(logicalIndex);
  const std::size_t subStateSize = std::size_t{1} << state().numLocalWires();
  const int32_t subStateIndex = static_cast<int32_t>(index / subStateSize);
  const auto localIndices = state().subStateIndices();
  const bool isLocal = std::find(localIndices.begin(), localIndices.end(),
                                 subStateIndex) != localIndices.end();
  std::complex<Scalar> result{};

  if (state().distributionType() !=
      CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS) {
    if (!isLocal)
      throw std::runtime_error("State amplitude is not locally accessible.");
    state().getState(&result, index, index + 1);
    state().synchronize();
    return {result.real(), result.imag()};
  }

  const auto communicator = state().communicator();
  if (!communicator)
    throw std::runtime_error("Distributed state has no communicator.");
  const std::size_t rankCount = static_cast<std::size_t>(communicator->size());
  std::vector<int32_t> gathered(localIndices.size() * rankCount);
  communicator->allGather(localIndices.data(), gathered.data(),
                          checkedCollectiveCount(localIndices.size()),
                          CUDA_R_32I);
  const auto position =
      std::find(gathered.begin(), gathered.end(), subStateIndex);
  if (position == gathered.end())
    throw std::runtime_error("Unable to locate distributed state amplitude.");
  const int32_t owner = static_cast<int32_t>(
      std::distance(gathered.begin(), position) / localIndices.size());
  if (isLocal)
    state().getState(&result, index, index + 1);
  // Ex synchronization includes an MPI barrier, so every rank must enter it
  // even though only the owning rank queued a one-element transfer.
  state().synchronize();
  communicator->broadcast(&result, 1, complexDataType<Scalar>(), owner);
  return {result.real(), result.imag()};
}

template <typename Scalar>
std::complex<double> CuStateVecSimulationState<Scalar>::getAmplitude(
    const std::vector<int> &basisState) {
  if (basisState.size() != getNumQubits())
    throw std::invalid_argument("Invalid basis-state width.");
  if (std::any_of(basisState.begin(), basisState.end(),
                  [](int bit) { return bit != 0 && bit != 1; }))
    throw std::invalid_argument("Basis-state values must be zero or one.");
  // Convert the logical basis state to its state-vector index. Wire ordering
  // is applied separately when locating the physical sub-state.
  const std::size_t logicalIndex = std::accumulate(
      basisState.rbegin(), basisState.rend(), std::size_t{0},
      [](std::size_t value, int bit) { return (value << 1) | bit; });
  return amplitudeAt(logicalIndex);
}

template <typename Scalar>
void CuStateVecSimulationState<Scalar>::dump(std::ostream &stream) const {
  if (!m_state && !m_scalarDevicePtr) {
    stream << "SV: nullptr\n";
    return;
  }
  // Print at most `maxPrinted` amplitudes so a large state does not flood the
  // stream.
  constexpr std::size_t maxPrinted = 100;
  const std::size_t total = numElements();
  const std::size_t count = std::min(total, maxPrinted);
  std::vector<std::complex<Scalar>> host(count);
  if (!m_scalarDevicePtr && state().distributionType() !=
                                CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS) {
    normalizeWireOrdering();
    state().getState(host.data(), 0, count);
    state().synchronize();
  } else {
    // Scalar (one element) or distributed (gathered across ranks): copyToHost
    // materializes the whole state, then keep the leading `count`.
    std::vector<std::complex<Scalar>> full(total);
    copyToHost(full.data(), total);
    std::copy_n(full.begin(), count, host.begin());
  }
  stream << "SV: [";
  for (std::size_t index = 0; index < count; ++index) {
    const auto &amplitude = host[index];
    stream << '(' << amplitude.real() << ',' << amplitude.imag() << ')';
    if (index + 1 != count)
      stream << ", ";
  }
  if (count < total)
    stream << ", ... (" << total << " amplitudes total)";
  stream << "]\n";
}

template <typename Scalar>
SimulationState::precision
CuStateVecSimulationState<Scalar>::getPrecision() const {
  if (!m_state && !m_scalarDevicePtr)
    throw std::runtime_error("Cannot query precision of a destroyed state.");
  if constexpr (std::is_same_v<Scalar, float>)
    return precision::fp32;
  return precision::fp64;
}

template <typename Scalar>
void CuStateVecSimulationState<Scalar>::destroyState() {
  m_state.reset();
  if (m_scalarDevicePtr)
    cudaFree(std::exchange(m_scalarDevicePtr, nullptr));
}

template <typename Scalar>
SimulationState::Tensor
CuStateVecSimulationState<Scalar>::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx != 0)
    throw std::out_of_range("Invalid cuStateVec tensor index.");
  if (m_scalarDevicePtr)
    return {m_scalarDevicePtr, {1}, getPrecision()};
  if (state().numMigrationWires() != 0)
    throw std::runtime_error(
        "Tensor access is not supported for a migrated state.");
  normalizeWireOrdering();
  const auto indices = state().deviceSubStateIndices();
  if (indices.empty())
    throw std::runtime_error("No device sub-state vector is available.");
  const auto resource = state().deviceSubStateVector(indices.front());
  return {const_cast<void *>(resource.data),
          {std::size_t{1} << state().numLocalWires()},
          getPrecision()};
}

template <typename Scalar>
std::vector<SimulationState::Tensor>
CuStateVecSimulationState<Scalar>::getTensors() const {
  return {getTensor()};
}

template <typename Scalar>
std::complex<double> CuStateVecSimulationState<Scalar>::operator()(
    std::size_t tensorIdx, const std::vector<std::size_t> &indices) {
  if (tensorIdx != 0 || indices.size() != 1 || indices.front() >= numElements())
    throw std::out_of_range("Invalid cuStateVec state element index.");
  return amplitudeAt(indices.front());
}

template <typename Scalar>
void CuStateVecSimulationState<Scalar>::toHost(
    std::complex<double> *data, std::size_t numElementsRequested) const {
  if constexpr (std::is_same_v<Scalar, float>)
    throw std::invalid_argument(
        "FP32 simulation state cannot be copied to an FP64 buffer.");
  else
    copyToHost(data, numElementsRequested);
}

template <typename Scalar>
void CuStateVecSimulationState<Scalar>::toHost(
    std::complex<float> *data, std::size_t numElementsRequested) const {
  if constexpr (std::is_same_v<Scalar, double>)
    throw std::invalid_argument(
        "FP64 simulation state cannot be copied to an FP32 buffer.");
  else
    copyToHost(data, numElementsRequested);
}

template <typename Scalar>
const CuStateVecState<Scalar> &
CuStateVecSimulationState<Scalar>::state() const {
  if (m_scalarDevicePtr)
    throw std::runtime_error(
        "A zero-qubit simulation state has no cuStateVecEx descriptor.");
  if (!m_state)
    throw std::runtime_error("The simulation state has been destroyed.");
  return *m_state;
}

template <typename Scalar>
std::unique_ptr<CuStateVecSimulationState<Scalar>>
CuStateVecSimulationState<Scalar>::create(std::size_t size, const void *data,
                                          bool allowFp32Emulation) {
  const int32_t numWires = static_cast<int32_t>(checkedQubitCount(size));
  if (!data)
    throw std::invalid_argument("State-vector data pointer cannot be null.");
  if (size == 1) {
    void *scalarDevicePtr = nullptr;
    HANDLE_CUDA_ERROR(
        cudaMalloc(&scalarDevicePtr, sizeof(std::complex<Scalar>)));
    auto result = std::unique_ptr<CuStateVecSimulationState<Scalar>>(
        new CuStateVecSimulationState<Scalar>(scalarDevicePtr));
    HANDLE_CUDA_ERROR(cudaMemcpy(scalarDevicePtr, data,
                                 sizeof(std::complex<Scalar>),
                                 cudaMemcpyDefault));
    return result;
  }

  int32_t device = 0;
  HANDLE_CUDA_ERROR(cudaGetDevice(&device));
  auto state = CuStateVecState<Scalar>::createSingleDevice(
      numWires, numWires, device, allowFp32Emulation);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, numWires);
  if (!state.setStateFromDevicePointer(data, size)) {
    state.setState(static_cast<const std::complex<Scalar> *>(data), 0, size);
    state.synchronize();
  }
  return std::make_unique<CuStateVecSimulationState<Scalar>>(std::move(state));
}

template <typename Scalar>
std::unique_ptr<SimulationState>
CuStateVecSimulationState<Scalar>::createFromSizeAndPtr(std::size_t size,
                                                        void *data,
                                                        std::size_t) {
  return create(size, data, true);
}

template class CuStateVecSimulationState<float>;
template class CuStateVecSimulationState<double>;

} // namespace cudaq::cusv
