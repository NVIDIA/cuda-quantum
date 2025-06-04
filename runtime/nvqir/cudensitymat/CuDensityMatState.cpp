/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "CuDensityMatState.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include "common/EigenDense.h"
#include "common/Logger.h"
#include "cudaq/utils/cudaq_utils.h"
namespace cudaq {

std::size_t CuDensityMatState::getNumQubits() const {
  if (!is_initialized())
    throw std::runtime_error("[CuDensityMatState] Get number of qubits for an "
                             "uninitiated state is not supported.");

  if (std::any_of(hilbertSpaceDims.begin(), hilbertSpaceDims.end(),
                  [](auto dim) { return dim != 2; }))
    throw std::runtime_error("[CuDensityMatState] Get number of qubits is only "
                             "supported on qubit (2-level) systems");
  return hilbertSpaceDims.size();
}

std::complex<double>
CuDensityMatState::overlap(const cudaq::SimulationState &other) {
  if (getTensor().extents != other.getTensor().extents)
    throw std::runtime_error("[CuDensityMatState] overlap error - other state "
                             "dimension not equal to this state dimension.");

  if (other.getPrecision() != getPrecision())
    throw std::runtime_error(
        "[CuDensityMatState] overlap error - precision mismatch.");

  if (!isDensityMatrix) {
    Eigen::VectorXcd state(dimension);
    const auto size = dimension;
    HANDLE_CUDA_ERROR(cudaMemcpy(state.data(), devicePtr,
                                 size * sizeof(std::complex<double>),
                                 cudaMemcpyDeviceToHost));

    Eigen::VectorXcd otherState(dimension);
    HANDLE_CUDA_ERROR(cudaMemcpy(otherState.data(), other.getTensor().data,
                                 size * sizeof(std::complex<double>),
                                 cudaMemcpyDeviceToHost));
    return std::abs(std::inner_product(
        state.begin(), state.end(), otherState.begin(),
        std::complex<double>{0., 0.}, [](auto a, auto b) { return a + b; },
        [](auto a, auto b) { return a * std::conj(b); }));
  }

  // FIXME: implement this in GPU memory
  Eigen::MatrixXcd state(dimension, dimension);
  const auto size = dimension * dimension;
  HANDLE_CUDA_ERROR(cudaMemcpy(state.data(), devicePtr,
                               size * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));

  Eigen::MatrixXcd otherState(dimension, dimension);
  HANDLE_CUDA_ERROR(cudaMemcpy(otherState.data(), other.getTensor().data,
                               size * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));

  return (state.adjoint() * otherState).trace();
}

std::complex<double>
CuDensityMatState::getAmplitude(const std::vector<int> &basisState) {
  throw std::runtime_error(
      "[CuDensityMatState] getAmplitude by basis states is not supported. "
      "Please use direct indexing access instead.");
}

// Dump the state to the given output stream
void CuDensityMatState::dump(std::ostream &os) const {
  // get state data from device to print
  const auto dim =
      isDensityMatrix ? std::size_t(std::sqrt(dimension)) : dimension;
  Eigen::MatrixXcd state(dim, isDensityMatrix ? dim : 1);
  const auto size = state.size();
  HANDLE_CUDA_ERROR(cudaMemcpy(state.data(), devicePtr,
                               size * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));
  os << state << std::endl;
}

std::unique_ptr<SimulationState>
CuDensityMatState::createFromSizeAndPtr(std::size_t size, void *dataPtr,
                                        std::size_t type) {
  bool isDm = false;
  if (type == cudaq::detail::variant_index<cudaq::state_data,
                                           cudaq::TensorStateData>()) {
    if (size != 1)
      throw std::runtime_error("[CuDensityMatState]: createFromSizeAndPtr "
                               "expects a single tensor");
    auto *casted =
        reinterpret_cast<cudaq::TensorStateData::value_type *>(dataPtr);

    auto [ptr, extents] = casted[0];
    if (extents.size() > 2)
      throw std::runtime_error("[CuDensityMatState]: createFromSizeAndPtr only "
                               "accept 1D or 2D arrays");

    isDm = extents.size() == 2;
    size = std::reduce(extents.begin(), extents.end(), 1, std::multiplies());
    dataPtr = const_cast<void *>(ptr);
  }
  std::complex<double> *devicePtr = static_cast<std::complex<double> *>(
      cudaq::dynamics::DeviceAllocator::allocate(size *
                                                 sizeof(std::complex<double>)));
  HANDLE_CUDA_ERROR(cudaMemcpy(devicePtr, dataPtr,
                               size * sizeof(std::complex<double>),
                               cudaMemcpyDefault));
  // printf("Created CuDensityMatState ptr %p\n", devicePtr);
  return std::make_unique<CuDensityMatState>(size, devicePtr);
}

// Return the tensor at the given index. Throws
// for an invalid tensor index.
cudaq::SimulationState::Tensor
CuDensityMatState::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx != 0)
    throw std::runtime_error(
        "CuDensityMatState state only supports a single tensor");

  const std::size_t dim = isDensityMatrix
                              ? static_cast<std::size_t>(std::sqrt(dimension))
                              : dimension;
  const std::vector<std::size_t> extents =
      isDensityMatrix ? std::vector<std::size_t>{dim, dim}
                      : std::vector<std::size_t>{dim};
  return Tensor{devicePtr, extents, precision::fp64};
}

std::complex<double>
CuDensityMatState::operator()(std::size_t tensorIdx,
                              const std::vector<std::size_t> &indices) {
  const auto extractValue = [&](std::size_t idx) {
    std::complex<double> value;
    HANDLE_CUDA_ERROR(cudaMemcpy(
        &value, reinterpret_cast<std::complex<double> *>(devicePtr) + idx,
        sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    return value;
  };

  if (tensorIdx != 0)
    throw std::runtime_error(
        "CuDensityMatState state only supports a single tensor");
  if (isDensityMatrix) {
    if (indices.size() != 2)
      throw std::runtime_error("CuDensityMatState holding a density matrix "
                               "supports only 2-dimensional indices");
    if (indices[0] >= dimension || indices[1] >= dimension)
      throw std::runtime_error("CuDensityMatState indices out of range");
    return extractValue(indices[0] * dimension + indices[1]);
  }
  if (indices.size() != 1)
    throw std::runtime_error(
        "CuDensityMatState holding a state vector supports "
        "only 1-dimensional indices");
  if (indices[0] >= dimension)
    throw std::runtime_error("Index out of bounds");
  return extractValue(indices[0]);
}

// Copy the state device data to the user-provided host data pointer.
void CuDensityMatState::toHost(std::complex<double> *userData,
                               std::size_t numElements) const {
  if (numElements != dimension)
    throw std::runtime_error(
        fmt::format("Number of elements in user data does not match "
                    "the size of the state: provided {}, expected {}.",
                    numElements, dimension));

  HANDLE_CUDA_ERROR(cudaMemcpy(userData, devicePtr,
                               numElements * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));
}

// Copy the state device data to the user-provided host data pointer.
void CuDensityMatState::toHost(std::complex<float> *userData,
                               std::size_t numElements) const {
  throw std::runtime_error(
      "CuDensityMatState: Data type mismatches - expecting "
      "double-precision array.");
}

// Free the device data.
void CuDensityMatState::destroyState() {
  if (cudmState) {
    cudensitymatDestroyState(cudmState);
    cudmState = nullptr;
  }
  if (devicePtr != nullptr) {
    cudaq::dynamics::DeviceAllocator::free(devicePtr);
    devicePtr = nullptr;
    dimension = 0;
    isDensityMatrix = false;
  }
}

static size_t
calculate_state_vector_size(const std::vector<int64_t> &hilbertSpaceDims) {
  return std::accumulate(hilbertSpaceDims.begin(), hilbertSpaceDims.end(), 1,
                         std::multiplies<>());
}

static size_t
calculate_density_matrix_size(const std::vector<int64_t> &hilbertSpaceDims) {
  size_t vectorSize = calculate_state_vector_size(hilbertSpaceDims);
  return vectorSize * vectorSize;
}

CuDensityMatState::CuDensityMatState(std::size_t size, void *ptr)
    : devicePtr(ptr), dimension(size),
      cudmHandle(dynamics::Context::getCurrentContext()->getHandle()) {
  if (size == 0)
    throw std::invalid_argument("Zero-length state is not allowed.");
}

std::unique_ptr<CuDensityMatState> CuDensityMatState::createInitialState(
    cudensitymatHandle_t handle, InitialState initial_state,
    const cudaq::dimension_map &dimensions, bool createDensityMatrix) {
  auto state = std::make_unique<CuDensityMatState>();
  state->cudmHandle = handle;
  std::size_t totalDim = 1;
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    const auto iter = dimensions.find(i);
    if (iter == dimensions.end())
      throw std::runtime_error(fmt::format(
          "Unable to find dimension of sub-system {} in the dimension map {}",
          i, dimensions));
    state->hilbertSpaceDims.emplace_back(iter->second);
    totalDim *= iter->second;
  }
  const cudensitymatStatePurity_t purity = createDensityMatrix
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  state->isDensityMatrix = createDensityMatrix;

  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      state->cudmHandle, purity,
      static_cast<int32_t>(state->hilbertSpaceDims.size()),
      state->hilbertSpaceDims.data(), 1, CUDA_C_64F, &state->cudmState));

  std::size_t storageSize;
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      state->cudmHandle, state->cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  const std::size_t stateVolume =
      storageSize / sizeof(std::complex<double>); // quantum state tensor volume
  // (number of elements)
  state->dimension = stateVolume;
  switch (initial_state) {
  case InitialState::ZERO: {
    const bool isFirstStateSegment = [&]() {
      if (stateVolume == totalDim || stateVolume == totalDim * totalDim)
        return true;

      int32_t numComponents = 0;
      HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(
          state->cudmHandle, state->cudmState, &numComponents));
      assert(numComponents == 1);
      int32_t numModes{0};
      int32_t stateComponentGlobalId{-1};
      int32_t batchModeLocation{-1};
      HANDLE_CUDM_ERROR(cudensitymatStateGetComponentNumModes(
          state->cudmHandle, state->cudmState, /*stateComponentLocalId=*/0,
          &stateComponentGlobalId, &numModes, &batchModeLocation));
      std::vector<int64_t> stateComponentModeExtents(numModes);
      std::vector<int64_t> stateComponentModeOffsets(numModes);

      HANDLE_CUDM_ERROR(cudensitymatStateGetComponentInfo(
          state->cudmHandle, state->cudmState, /*stateComponentLocalId=*/0,
          &stateComponentGlobalId, &numModes, stateComponentModeExtents.data(),
          stateComponentModeOffsets.data()));
      // All the offsets are zero
      return std::all_of(stateComponentModeOffsets.cbegin(),
                         stateComponentModeOffsets.cend(),
                         [](int64_t i) { return i == 0; });
    }();

    state->devicePtr = cudaq::dynamics::DeviceAllocator::allocate(storageSize);
    HANDLE_CUDA_ERROR(cudaMemset(state->devicePtr, 0, storageSize));
    if (isFirstStateSegment) {
      // Set the first element to 1.0
      constexpr std::complex<double> oneVal = 1.0;
      HANDLE_CUDA_ERROR(cudaMemcpy(state->devicePtr, &oneVal,
                                   sizeof(std::complex<double>),
                                   cudaMemcpyDefault));
    }
    break;
  }
  case InitialState::UNIFORM: {
    const double factor = createDensityMatrix
                              ? static_cast<double>(totalDim)
                              : std::sqrt(static_cast<double>(totalDim));
    std::vector<std::complex<double>> uniformState(stateVolume, 1.0 / factor);
    state->devicePtr = cudaq::dynamics::createArrayGpu(uniformState);
    break;
  }
  default:
    __builtin_unreachable();
    break;
  }
  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      state->cudmHandle, state->cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({state->devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
  return state;
}

CuDensityMatState CuDensityMatState::zero_like(const CuDensityMatState &other) {
  CuDensityMatState state;
  state.cudmHandle = other.cudmHandle;
  state.hilbertSpaceDims = other.hilbertSpaceDims;
  state.dimension = other.dimension;
  state.isDensityMatrix = other.isDensityMatrix;
  state.batchSize = other.batchSize;
  const size_t dataSize = state.dimension * sizeof(std::complex<double>);
  state.devicePtr = cudaq::dynamics::DeviceAllocator::allocate(dataSize);
  HANDLE_CUDA_ERROR(cudaMemset(state.devicePtr, 0, dataSize));
  const cudensitymatStatePurity_t purity = state.isDensityMatrix
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      state.cudmHandle, purity,
      static_cast<int32_t>(state.hilbertSpaceDims.size()),
      state.hilbertSpaceDims.data(), state.batchSize, CUDA_C_64F,
      &state.cudmState));

  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      state.cudmHandle, state.cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({state.devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({dataSize})
          .data())); // size of the GPU storage for the quantum state
  return state;
}

std::unique_ptr<CuDensityMatState>
CuDensityMatState::clone(const CuDensityMatState &other) {
  assert(other.is_initialized());
  CuDensityMatState *state = new CuDensityMatState;
  state->cudmHandle = other.cudmHandle;
  state->hilbertSpaceDims = other.hilbertSpaceDims;
  state->dimension = other.dimension;
  state->isDensityMatrix = other.isDensityMatrix;
  state->batchSize = other.batchSize;
  const size_t dataSize = state->dimension * sizeof(std::complex<double>);
  state->devicePtr = cudaq::dynamics::DeviceAllocator::allocate(dataSize);
  HANDLE_CUDA_ERROR(cudaMemcpy(state->devicePtr, other.devicePtr, dataSize,
                               cudaMemcpyDefault));
  const cudensitymatStatePurity_t purity = state->isDensityMatrix
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      state->cudmHandle, purity,
      static_cast<int32_t>(state->hilbertSpaceDims.size()),
      state->hilbertSpaceDims.data(), /*batchSize=*/state->batchSize,
      CUDA_C_64F, &state->cudmState));

  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      state->cudmHandle, state->cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({state->devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({dataSize})
          .data())); // size of the GPU storage for the quantum state
  return std::unique_ptr<CuDensityMatState>(state);
}

CuDensityMatState::CuDensityMatState(CuDensityMatState &&other) noexcept
    : isDensityMatrix(other.isDensityMatrix), dimension(other.dimension),
      devicePtr(other.devicePtr), cudmState(other.cudmState),
      cudmHandle(other.cudmHandle), hilbertSpaceDims(other.hilbertSpaceDims),
      batchSize(other.batchSize) {
  other.isDensityMatrix = false;
  other.dimension = 0;
  other.devicePtr = nullptr;
  other.batchSize = 1;
  other.cudmState = nullptr;
  other.cudmHandle = nullptr;
  other.hilbertSpaceDims.clear();
}

CuDensityMatState &
CuDensityMatState::operator=(CuDensityMatState &&other) noexcept {
  if (this != &other) {
    // Free existing resources
    if (cudmState)
      cudensitymatDestroyState(cudmState);

    if (devicePtr) {
      cudaq::dynamics::DeviceAllocator::free(devicePtr);
    }

    // Move data from other
    isDensityMatrix = other.isDensityMatrix;
    dimension = other.dimension;
    devicePtr = other.devicePtr;
    cudmState = other.cudmState;
    cudmHandle = other.cudmHandle;
    hilbertSpaceDims = std::move(other.hilbertSpaceDims);
    batchSize = other.batchSize;
    // Nullify other
    other.isDensityMatrix = false;
    other.dimension = 0;
    other.devicePtr = nullptr;

    other.cudmState = nullptr;
    other.batchSize = 1;
  }
  return *this;
}

CuDensityMatState::~CuDensityMatState() { destroyState(); }

bool CuDensityMatState::is_initialized() const { return cudmState != nullptr; }

bool cudaq::CuDensityMatState::is_density_matrix() const {
  if (!is_initialized())
    return false;

  return isDensityMatrix;
}

CuDensityMatState cudaq::CuDensityMatState::to_density_matrix() const {
  if (!is_initialized())
    throw std::runtime_error("State is not initialized.");

  if (is_density_matrix())
    throw std::runtime_error("State is already a density matrix.");

  if (batchSize > 1)
    throw std::runtime_error(
        "Conversion of a batched state to a density matrix is not supported.");

  const std::size_t vectorSize = calculate_state_vector_size(hilbertSpaceDims);
  const std::size_t expectedDensityMatrixSize = vectorSize * vectorSize;
  const std::size_t dmSizeBytes =
      expectedDensityMatrixSize * sizeof(std::complex<double>);

  CuDensityMatState dmState;
  dmState.devicePtr = cudaq::dynamics::DeviceAllocator::allocate(dmSizeBytes);
  dmState.isDensityMatrix = true;
  HANDLE_CUDA_ERROR(cudaMemset(dmState.devicePtr, 0, dmSizeBytes));
  dmState.dimension = expectedDensityMatrixSize;
  cuDoubleComplex scalar{1.0, 0.0};
  HANDLE_CUBLAS_ERROR(cublasZgerc(
      dynamics::Context::getCurrentContext()->getCublasHandle(), vectorSize,
      vectorSize, &scalar, reinterpret_cast<const cuDoubleComplex *>(devicePtr),
      1, reinterpret_cast<const cuDoubleComplex *>(devicePtr), 1,
      reinterpret_cast<cuDoubleComplex *>(dmState.devicePtr), vectorSize));
  dmState.initialize_cudm(cudmHandle, hilbertSpaceDims, batchSize);
  assert(dmState.is_initialized());
  assert(dmState.is_density_matrix());
  return dmState;
}

cudensitymatState_t cudaq::CuDensityMatState::get_impl() const {
  return cudmState;
}

void *cudaq::CuDensityMatState::get_device_pointer() const { return devicePtr; }

std::vector<int64_t> cudaq::CuDensityMatState::get_hilbert_space_dims() const {
  return hilbertSpaceDims;
}

cudensitymatHandle_t cudaq::CuDensityMatState::get_handle() const {
  return cudmHandle;
}

void CuDensityMatState::initialize_cudm(cudensitymatHandle_t handleToSet,
                                        const std::vector<int64_t> &dims,
                                        int64_t batchSize) {
  assert(!is_initialized());
  cudmHandle = handleToSet;
  hilbertSpaceDims = dims;
  size_t expectedDensityMatrixSize =
      calculate_density_matrix_size(hilbertSpaceDims);
  size_t expectedStateVectorSize =
      calculate_state_vector_size(hilbertSpaceDims);
  const int64_t totalDistributedDimension =
      cudaq::dynamics::getNumRanks() * dimension;
  if (dimension != batchSize * expectedDensityMatrixSize &&
      dimension != batchSize * expectedStateVectorSize &&
      totalDistributedDimension != batchSize * expectedDensityMatrixSize &&
      totalDistributedDimension != batchSize * expectedStateVectorSize) {
    throw std::invalid_argument("Invalid hilbertSpaceDims for the state data");
  }

  isDensityMatrix =
      (dimension == batchSize * expectedDensityMatrixSize ||
       totalDistributedDimension == batchSize * expectedDensityMatrixSize);
  const cudensitymatStatePurity_t purity = isDensityMatrix
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  this->batchSize = batchSize;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      cudmHandle, purity, static_cast<int32_t>(hilbertSpaceDims.size()),
      hilbertSpaceDims.data(), batchSize, CUDA_C_64F, &cudmState));

  std::size_t storageSize;
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      cudmHandle, cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  const std::size_t stateVolume =
      storageSize / sizeof(std::complex<double>); // quantum state tensor volume
                                                  // (number of elements)
  if (stateVolume < dimension) {
    int32_t numComponents = 0;
    HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(cudmHandle, cudmState,
                                                        &numComponents));
    assert(numComponents == 1);
    int32_t numModes{0};
    int32_t stateComponentGlobalId{-1};
    int32_t batchModeLocation{-1};
    HANDLE_CUDM_ERROR(cudensitymatStateGetComponentNumModes(
        cudmHandle, cudmState, /*stateComponentLocalId=*/0,
        &stateComponentGlobalId, &numModes, &batchModeLocation));
    std::vector<int64_t> stateComponentModeExtents(numModes);
    std::vector<int64_t> stateComponentModeOffsets(numModes);

    HANDLE_CUDM_ERROR(cudensitymatStateGetComponentInfo(
        cudmHandle, cudmState, /*stateComponentLocalId=*/0,
        &stateComponentGlobalId, &numModes, stateComponentModeExtents.data(),
        stateComponentModeOffsets.data()));

    dimension = stateVolume;
    int64_t startIdx = 0;
    int64_t accumulatedIdx = 1;
    for (int32_t i = 0; i < numModes; ++i) {
      accumulatedIdx *= stateComponentModeExtents[i];
      startIdx += (stateComponentModeOffsets[i] * accumulatedIdx);
    }
    if (startIdx > 0) {
      std::complex<double> *startPtr =
          static_cast<std::complex<double> *>(devicePtr) + startIdx;
      HANDLE_CUDA_ERROR(cudaMemcpy(devicePtr, startPtr,
                                   stateVolume * sizeof(std::complex<double>),
                                   cudaMemcpyDefault));
    }
  }
  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      cudmHandle, cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
}

void CuDensityMatState::accumulate_inplace(const CuDensityMatState &other,
                                           const std::complex<double> &coeff) {

  if (dimension != other.dimension)
    throw std::invalid_argument(
        fmt::format("State size mismatch for accumulate_inplace ({} vs {}).",
                    dimension, other.dimension));

  {
    cudaq::dynamics::PerfMetricScopeTimer metricTimer("cublasZaxpy");
    cuDoubleComplex scalar{coeff.real(), coeff.imag()};
    HANDLE_CUBLAS_ERROR(cublasZaxpy(
        dynamics::Context::getCurrentContext()->getCublasHandle(), dimension,
        &scalar, reinterpret_cast<const cuDoubleComplex *>(other.devicePtr), 1,
        reinterpret_cast<cuDoubleComplex *>(devicePtr), 1));
  }
}

CuDensityMatState &
cudaq::CuDensityMatState::operator+=(const CuDensityMatState &other) {
  if (dimension != other.dimension)
    throw std::invalid_argument(
        fmt::format("State size mismatch for addition ({} vs {}).", dimension,
                    other.dimension));

  accumulate_inplace(other);
  return *this;
}

CuDensityMatState &
cudaq::CuDensityMatState::operator*=(const std::complex<double> &scalar) {
  HANDLE_CUBLAS_ERROR(
      cublasZscal(dynamics::Context::getCurrentContext()->getCublasHandle(),
                  dimension, reinterpret_cast<const cuDoubleComplex *>(&scalar),
                  reinterpret_cast<cuDoubleComplex *>(devicePtr), 1));

  return *this;
}

std::vector<CuDensityMatState *>
CuDensityMatState::convertStateVecToDensityMatrix(
    const std::vector<CuDensityMatState *> svStates, int64_t dmSize) {
  std::vector<CuDensityMatState *> dmStates;
  const auto dmSizeBytes = dmSize * sizeof(std::complex<double>);
  for (auto *stateVecState : svStates) {
    auto cudmState = new CuDensityMatState();
    cudmState->devicePtr =
        cudaq::dynamics::DeviceAllocator::allocate(dmSizeBytes);
    cudmState->isDensityMatrix = true;
    HANDLE_CUDA_ERROR(cudaMemset(cudmState->devicePtr, 0, dmSizeBytes));
    cudmState->dimension = dmSize;
    cuDoubleComplex scalar{1.0, 0.0};
    HANDLE_CUBLAS_ERROR(cublasZgerc(
        dynamics::Context::getCurrentContext()->getCublasHandle(),
        stateVecState->dimension, stateVecState->dimension, &scalar,
        reinterpret_cast<const cuDoubleComplex *>(stateVecState->devicePtr), 1,
        reinterpret_cast<const cuDoubleComplex *>(stateVecState->devicePtr), 1,
        reinterpret_cast<cuDoubleComplex *>(cudmState->devicePtr),
        stateVecState->dimension));
    dmStates.emplace_back(cudmState);
  }
  return dmStates;
}

void CuDensityMatState::distributeBatchedStateData(
    CuDensityMatState &batchedState,
    const std::vector<CuDensityMatState *> inputStates,
    int64_t singleStateDimension) {
  int32_t numComponents = 0;
  HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(
      batchedState.cudmHandle, batchedState.cudmState, &numComponents));
  assert(numComponents == 1);
  int32_t numModes{0};
  int32_t stateComponentGlobalId{-1};
  int32_t batchModeLocation{-1};
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentNumModes(
      batchedState.cudmHandle, batchedState.cudmState,
      /*stateComponentLocalId=*/0, &stateComponentGlobalId, &numModes,
      &batchModeLocation));
  std::vector<int64_t> stateComponentModeExtents(numModes);
  std::vector<int64_t> stateComponentModeOffsets(numModes);

  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentInfo(
      batchedState.cudmHandle, batchedState.cudmState,
      /*stateComponentLocalId=*/0, &stateComponentGlobalId, &numModes,
      stateComponentModeExtents.data(), stateComponentModeOffsets.data()));
  int64_t startIdx = 0;
  int64_t accumulatedIdx = 1;
  for (int32_t i = 0; i < numModes; ++i) {
    accumulatedIdx *= stateComponentModeExtents[i];
    startIdx += (stateComponentModeOffsets[i] * accumulatedIdx);
  }
  const int batchIdx = startIdx / singleStateDimension;
  const int64_t stateVolume = batchedState.dimension;
  const int numStatesPerGpu = stateVolume / singleStateDimension;
  if (batchIdx * singleStateDimension != startIdx)
    throw std::runtime_error(
        "Batched state cannot be evenly distributed across available GPUs");
  for (int i = 0; i < numStatesPerGpu; ++i) {
    auto *sourcePtr = inputStates[i + batchIdx]->devicePtr;
    std::complex<double> *destPtr =
        static_cast<std::complex<double> *>(batchedState.devicePtr) +
        i * singleStateDimension;
    HANDLE_CUDA_ERROR(cudaMemcpy(
        destPtr, sourcePtr, singleStateDimension * sizeof(std::complex<double>),
        cudaMemcpyDefault));
  }
}

std::unique_ptr<CuDensityMatState> CuDensityMatState::createBatchedState(
    cudensitymatHandle_t handle,
    const std::vector<CuDensityMatState *> initial_states,
    const std::vector<int64_t> &dimensions, bool createDensityState) {
  if (initial_states.size() < 2)
    throw std::invalid_argument(
        "Batched state needs more than 1 input states.");
  const auto firstStateDimension = initial_states[0]->dimension;
  if (std::any_of(initial_states.begin(), initial_states.end(),
                  [firstStateDimension](CuDensityMatState *state) {
                    return state->dimension != firstStateDimension;
                  }))
    throw std::invalid_argument("All states must have the same dimension");

  const size_t expectedDensityMatrixSize =
      calculate_density_matrix_size(dimensions);
  const size_t expectedStateVectorSize =
      calculate_state_vector_size(dimensions);
  if (firstStateDimension != expectedDensityMatrixSize &&
      firstStateDimension != expectedStateVectorSize)
    throw std::invalid_argument("Invalid hilbertSpaceDims for the state data");

  const bool isDm = firstStateDimension == expectedDensityMatrixSize;
  // These are state vectors but we need density matrices (e.g., with collapsed
  // operators)
  if (!isDm && createDensityState) {
    std::vector<CuDensityMatState *> initialDensityMatrixStates =
        convertStateVecToDensityMatrix(initial_states,
                                       expectedDensityMatrixSize);
    auto batchedDmState = createBatchedState(handle, initialDensityMatrixStates,
                                             dimensions, createDensityState);
    for (auto *dmState : initialDensityMatrixStates) {
      delete dmState;
    }
    return batchedDmState;
  }

  auto cudmState = std::make_unique<CuDensityMatState>();
  cudmState->cudmHandle = handle;
  cudmState->hilbertSpaceDims = dimensions;
  cudmState->batchSize = initial_states.size();
  cudmState->isDensityMatrix = isDm;
  const cudensitymatStatePurity_t purity = cudmState->isDensityMatrix
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      cudmState->cudmHandle, purity,
      static_cast<int32_t>(cudmState->hilbertSpaceDims.size()),
      cudmState->hilbertSpaceDims.data(), cudmState->batchSize, CUDA_C_64F,
      &cudmState->cudmState));

  std::size_t storageSize;
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      cudmState->cudmHandle, cudmState->cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  const std::size_t stateVolume =
      storageSize / sizeof(std::complex<double>); // quantum state tensor volume
                                                  // (number of elements)
  cudmState->devicePtr =
      cudaq::dynamics::DeviceAllocator::allocate(storageSize);
  cudmState->dimension = stateVolume;

  if (stateVolume < cudmState->batchSize * firstStateDimension) {
    // The batched state is distributed.
    distributeBatchedStateData(*cudmState, initial_states, firstStateDimension);
  } else {
    std::complex<double> *destPtr =
        static_cast<std::complex<double> *>(cudmState->devicePtr);
    // The batched state is an aggregated buffer.
    for (auto *initial_state : initial_states) {
      auto *sourcePtr = initial_state->devicePtr;
      HANDLE_CUDA_ERROR(
          cudaMemcpy(destPtr, sourcePtr,
                     firstStateDimension * sizeof(std::complex<double>),
                     cudaMemcpyDefault));
      destPtr += firstStateDimension;
    }
  }
  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      cudmState->cudmHandle, cudmState->cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({cudmState->devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
  return cudmState;
}

std::vector<CuDensityMatState *>
CuDensityMatState::splitBatchedState(CuDensityMatState &batchedState) {
  if (!batchedState.is_initialized()) {
    throw std::runtime_error("Uninitialized state");
  }

  if (batchedState.batchSize <= 1) {
    throw std::runtime_error("Input is not a batched state");
  }
  const int64_t stateSize = batchedState.dimension / batchedState.batchSize;
  std::complex<double> *ptr =
      static_cast<std::complex<double> *>(batchedState.devicePtr);
  std::vector<CuDensityMatState *> splitStates;
  for (int i = 0; i < batchedState.batchSize; ++i) {
    splitStates.emplace_back(
        new CuDensityMatState(stateSize, ptr + i * stateSize));
  }
  return splitStates;
}
} // namespace cudaq
