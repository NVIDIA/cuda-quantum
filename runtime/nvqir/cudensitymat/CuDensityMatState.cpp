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
#include "common/EigenDense.h"
#include "common/Logger.h"
#include "cudaq/utils/cudaq_utils.h"

namespace cudaq {

std::complex<double>
CuDensityMatState::overlap(const cudaq::SimulationState &other) {
  if (getTensor().extents != other.getTensor().extents)
    throw std::runtime_error("[CuDensityMatState] overlap error - other state "
                             "dimension not equal to this state dimension.");

  if (other.getPrecision() != getPrecision()) {
    throw std::runtime_error(
        "[CuDensityMatState] overlap error - precision mismatch.");
  }

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
  Eigen::MatrixXcd state(dimension, isDensityMatrix ? dimension : 1);
  const auto size = isDensityMatrix ? dimension * dimension : dimension;
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

  std::complex<double> *devicePtr = nullptr;

  HANDLE_CUDA_ERROR(
      cudaMalloc((void **)&devicePtr, size * sizeof(std::complex<double>)));
  HANDLE_CUDA_ERROR(cudaMemcpy(devicePtr, dataPtr,
                               size * sizeof(std::complex<double>),
                               cudaMemcpyDefault));
  // printf("Created CuDensityMatState ptr %p\n", devicePtr);
  return std::make_unique<CuDensityMatState>(size, devicePtr, isDm);
}

// Return the tensor at the given index. Throws
// for an invalid tensor index.
cudaq::SimulationState::Tensor
CuDensityMatState::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx != 0) {
    throw std::runtime_error(
        "CuDensityMatState state only supports a single tensor");
  }
  const std::vector<std::size_t> extents =
      isDensityMatrix ? std::vector<std::size_t>{dimension, dimension}
                      : std::vector<std::size_t>{dimension};
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
  if (numElements != dimension * (isDensityMatrix ? dimension : 1)) {
    throw std::runtime_error("Number of elements in user data does not match "
                             "the size of the state");
  }
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
    HANDLE_CUDA_ERROR(cudaFree(devicePtr));
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

CuDensityMatState::CuDensityMatState(
    cudensitymatHandle_t handle,
    const std::vector<std::complex<double>> &rawData,
    const std::vector<int64_t> &dims)
    : cudmHandle(handle), dimension(rawData.size()), cudmState(nullptr),
      hilbertSpaceDims(dims) {
  if (rawData.empty()) {
    throw std::invalid_argument("Raw data cannot be empty.");
  }
  // Allocate device memory
  size_t dataSize = rawData.size() * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(
      cudaMalloc(reinterpret_cast<void **>(&devicePtr), dataSize));

  // Copy data from host to device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(devicePtr, rawData.data(), dataSize, cudaMemcpyHostToDevice));

  // Determine if this is a denisty matrix or state vector
  size_t rawDataSize = rawData.size();
  size_t expectedDensityMatrixSize =
      calculate_density_matrix_size(hilbertSpaceDims);
  size_t expectedStateVectorSize =
      calculate_state_vector_size(hilbertSpaceDims);

  if (rawDataSize != expectedDensityMatrixSize &&
      rawDataSize != expectedStateVectorSize) {
    throw std::invalid_argument(
        "Invalid rawData size for the given Hilbert space dimensions.");
  }

  cudensitymatStatePurity_t purity;

  if (rawDataSize == expectedDensityMatrixSize) {
    purity = CUDENSITYMAT_STATE_PURITY_MIXED;
  } else if (rawDataSize == expectedStateVectorSize) {
    purity = CUDENSITYMAT_STATE_PURITY_PURE;
  }

  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      cudmHandle, purity, static_cast<int32_t>(hilbertSpaceDims.size()),
      hilbertSpaceDims.data(), 1, CUDA_C_64F, &cudmState));

  // Retrieve the number of state components
  int32_t numStateComponents;
  HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(cudmHandle, cudmState,
                                                      &numStateComponents));

  // Retrieve the storage size for each component
  std::vector<size_t> componentBufferSizes(numStateComponents);
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      cudmHandle, cudmState, numStateComponents, componentBufferSizes.data()));

  // Validate device memory
  size_t totalSize = std::accumulate(componentBufferSizes.begin(),
                                     componentBufferSizes.end(), 0);
  if (totalSize > rawData.size() * sizeof(std::complex<double>)) {
    throw std::invalid_argument(
        "Device memory size is insufficient to cover all components.");
  }

  // Attach storage for using device memory (devicePtr)
  std::vector<void *> componentBuffers(numStateComponents);
  size_t offset = 0;
  for (int32_t i = 0; i < numStateComponents; i++) {
    componentBuffers[i] = static_cast<void *>(
        static_cast<std::complex<double> *>(devicePtr) + offset);
    offset += componentBufferSizes[i] / sizeof(std::complex<double>);
  }

  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      cudmHandle, cudmState, numStateComponents, componentBuffers.data(),
      componentBufferSizes.data()));
}

CuDensityMatState::CuDensityMatState(cudensitymatHandle_t handle,
                                     const CuDensityMatState &simState,
                                     const std::vector<int64_t> &dims)
    : cudmHandle(handle), hilbertSpaceDims(dims) {

  const bool isDensityMat =
      simState.dimension == calculate_density_matrix_size(hilbertSpaceDims);
  dimension = simState.dimension;

  const size_t dataSize = dimension * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(
      cudaMalloc(reinterpret_cast<void **>(&devicePtr), dataSize));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(devicePtr, simState.devicePtr, dataSize, cudaMemcpyDefault));

  const cudensitymatStatePurity_t purity = isDensityMat
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      cudmHandle, purity, static_cast<int32_t>(hilbertSpaceDims.size()),
      hilbertSpaceDims.data(), 1, CUDA_C_64F, &cudmState));

  // Query the size of the quantum state storage
  std::size_t storageSize{0}; // only one storage component (tensor) is needed
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      cudmHandle, cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  const std::size_t stateVolume =
      storageSize / sizeof(std::complex<double>); // quantum state tensor volume
                                                  // (number of elements)
  assert(stateVolume == dimension);
  // std::cout << "Quantum state storage size (bytes) = " << storageSize
  //           << std::endl;

  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      cudmHandle, cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
}

CuDensityMatState CuDensityMatState::zero_like(const CuDensityMatState &other) {
  CuDensityMatState state;
  state.cudmHandle = other.cudmHandle;
  state.hilbertSpaceDims = other.hilbertSpaceDims;
  state.dimension = other.dimension;
  const size_t dataSize = state.dimension * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(
      cudaMalloc(reinterpret_cast<void **>(&state.devicePtr), dataSize));
  HANDLE_CUDA_ERROR(cudaMemset(state.devicePtr, 0, dataSize));

  const size_t expectedDensityMatrixSize =
      calculate_density_matrix_size(state.hilbertSpaceDims);
  const bool isDensityMat = expectedDensityMatrixSize == state.dimension;
  const cudensitymatStatePurity_t purity = isDensityMat
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      state.cudmHandle, purity,
      static_cast<int32_t>(state.hilbertSpaceDims.size()),
      state.hilbertSpaceDims.data(), 1, CUDA_C_64F, &state.cudmState));

  // Query the size of the quantum state storage
  std::size_t storageSize{0}; // only one storage component (tensor) is needed
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      state.cudmHandle, state.cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  const std::size_t stateVolume =
      storageSize / sizeof(std::complex<double>); // quantum state tensor volume
                                                  // (number of elements)
  assert(stateVolume == state.dimension);
  // std::cout << "Quantum state storage size (bytes) = " << storageSize
  //           << std::endl;

  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      state.cudmHandle, state.cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({state.devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
  return state;
}

CuDensityMatState CuDensityMatState::clone(const CuDensityMatState &other) {
  CuDensityMatState state;
  state.cudmHandle = other.cudmHandle;
  state.hilbertSpaceDims = other.hilbertSpaceDims;
  state.dimension = other.dimension;
  const size_t dataSize = state.dimension * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(
      cudaMalloc(reinterpret_cast<void **>(&state.devicePtr), dataSize));
  HANDLE_CUDA_ERROR(cudaMemcpy(state.devicePtr, other.devicePtr, dataSize,
                               cudaMemcpyDefault));

  const size_t expectedDensityMatrixSize =
      calculate_density_matrix_size(state.hilbertSpaceDims);
  const bool isDensityMat = expectedDensityMatrixSize == state.dimension;
  const cudensitymatStatePurity_t purity = isDensityMat
                                               ? CUDENSITYMAT_STATE_PURITY_MIXED
                                               : CUDENSITYMAT_STATE_PURITY_PURE;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      state.cudmHandle, purity,
      static_cast<int32_t>(state.hilbertSpaceDims.size()),
      state.hilbertSpaceDims.data(), 1, CUDA_C_64F, &state.cudmState));

  // Query the size of the quantum state storage
  std::size_t storageSize{0}; // only one storage component (tensor) is needed
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      state.cudmHandle, state.cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  const std::size_t stateVolume =
      storageSize / sizeof(std::complex<double>); // quantum state tensor volume
                                                  // (number of elements)
  assert(stateVolume == state.dimension);
  // std::cout << "Quantum state storage size (bytes) = " << storageSize
  //           << std::endl;

  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      state.cudmHandle, state.cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({state.devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
  return state;
}

CuDensityMatState::CuDensityMatState(CuDensityMatState &&other) noexcept
    : isDensityMatrix(other.isDensityMatrix), dimension(other.dimension),
      devicePtr(other.devicePtr), cudmState(other.cudmState),
      cudmHandle(other.cudmHandle), hilbertSpaceDims(other.hilbertSpaceDims) {
  other.isDensityMatrix = false;
  other.dimension = 0;
  other.devicePtr = nullptr;

  other.cudmState = nullptr;
  other.cudmHandle = nullptr;
  other.hilbertSpaceDims.clear();
}

CuDensityMatState &
CuDensityMatState::operator=(CuDensityMatState &&other) noexcept {
  if (this != &other) {
    // Free existing resources
    if (cudmState) {
      cudensitymatDestroyState(cudmState);
    }
    if (devicePtr) {
      cudaFree(devicePtr);
    }

    // Move data from other
    isDensityMatrix = other.isDensityMatrix;
    dimension = other.dimension;
    devicePtr = other.devicePtr;
    cudmState = other.cudmState;
    cudmHandle = other.cudmHandle;
    hilbertSpaceDims = std::move(other.hilbertSpaceDims);

    // Nullify other
    other.isDensityMatrix = false;
    other.dimension = 0;
    other.devicePtr = nullptr;

    other.cudmState = nullptr;
  }
  return *this;
}

CuDensityMatState::~CuDensityMatState() { destroyState(); }

bool CuDensityMatState::is_initialized() const { return cudmState != nullptr; }

bool cudaq::CuDensityMatState::is_density_matrix() const {
  if (!is_initialized()) {
    return false;
  }

  return dimension == calculate_density_matrix_size(hilbertSpaceDims);
}

CuDensityMatState cudaq::CuDensityMatState::to_density_matrix() const {
  if (!is_initialized()) {
    throw std::runtime_error("State is not initialized.");
  }

  if (is_density_matrix()) {
    throw std::runtime_error("State is already a density matrix.");
  }

  size_t vectorSize = calculate_state_vector_size(hilbertSpaceDims);
  std::vector<std::complex<double>> stateVecData(vectorSize);
  HANDLE_CUDA_ERROR(cudaMemcpy(stateVecData.data(), devicePtr,
                               dimension * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));
  size_t expectedDensityMatrixSize = vectorSize * vectorSize;
  std::vector<std::complex<double>> densityMatrix(expectedDensityMatrixSize);

  for (size_t i = 0; i < vectorSize; i++) {
    for (size_t j = 0; j < vectorSize; j++) {
      densityMatrix[i * vectorSize + j] =
          stateVecData[i] * std::conj(stateVecData[j]);
    }
  }

  return CuDensityMatState(cudmHandle, densityMatrix, hilbertSpaceDims);
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
                                        const std::vector<int64_t> &dims) {
  cudmHandle = handleToSet;
  hilbertSpaceDims = dims;
  size_t expectedDensityMatrixSize =
      calculate_density_matrix_size(hilbertSpaceDims);
  size_t expectedStateVectorSize =
      calculate_state_vector_size(hilbertSpaceDims);

  if (dimension != expectedDensityMatrixSize &&
      dimension != expectedStateVectorSize) {
    throw std::invalid_argument("Invalid hilbertSpaceDims for the state data");
  }

  const cudensitymatStatePurity_t purity =
      dimension == expectedDensityMatrixSize ? CUDENSITYMAT_STATE_PURITY_MIXED
                                             : CUDENSITYMAT_STATE_PURITY_PURE;

  HANDLE_CUDM_ERROR(cudensitymatCreateState(
      cudmHandle, purity, static_cast<int32_t>(hilbertSpaceDims.size()),
      hilbertSpaceDims.data(), 1, CUDA_C_64F, &cudmState));

  std::size_t storageSize;
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      cudmHandle, cudmState,
      1,              // only one storage component
      &storageSize)); // storage size in bytes
  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      cudmHandle, cudmState,
      1, // only one storage component (tensor)
      std::vector<void *>({devicePtr})
          .data(), // pointer to the GPU storage for the quantum state
      std::vector<std::size_t>({storageSize})
          .data())); // size of the GPU storage for the quantum state
}

CuDensityMatState
cudaq::CuDensityMatState::operator+(const CuDensityMatState &other) const {
  if (dimension != other.dimension) {
    throw std::invalid_argument("State size mismatch for addition.");
  }

  CuDensityMatState result = CuDensityMatState::clone(*this);

  result += other;

  return result;
}

CuDensityMatState &
cudaq::CuDensityMatState::operator+=(const CuDensityMatState &other) {
  if (dimension != other.dimension) {
    throw std::invalid_argument(
        fmt::format("State size mismatch for addition ({} vs {}).", dimension,
                    other.dimension));
  }

  // double scalingFactor = 1.0;
  // double *gpuScalingFactor;
  // cudaMalloc(reinterpret_cast<void **>(&gpuScalingFactor), sizeof(double));
  // cudaMemcpy(gpuScalingFactor, &scalingFactor, sizeof(double),
  //            cudaMemcpyHostToDevice);

  // HANDLE_CUDM_ERROR(cudensitymatStateComputeAccumulation(
  //     cudmHandle, other.get_impl(), cudmState, gpuScalingFactor, 0));

  // cudaFree(gpuScalingFactor);
  cuDoubleComplex scalar{1.0, 0.0};
  HANDLE_CUBLAS_ERROR(cublasZaxpy(
      dynamics::Context::getCurrentContext()->getCublasHandle(), dimension,
      &scalar, reinterpret_cast<const cuDoubleComplex *>(other.devicePtr), 1,
      reinterpret_cast<cuDoubleComplex *>(devicePtr), 1));
  return *this;
}

CuDensityMatState &
cudaq::CuDensityMatState::operator*=(const std::complex<double> &scalar) {
  // void *gpuScalar;
  // HANDLE_CUDA_ERROR(cudaMalloc(&gpuScalar, sizeof(std::complex<double>)));
  // HANDLE_CUDA_ERROR(cudaMemcpy(gpuScalar, &scalar,
  // sizeof(std::complex<double>),
  //                              cudaMemcpyHostToDevice));

  // HANDLE_CUDM_ERROR(
  //     cudensitymatStateComputeScaling(cudmHandle, cudmState, gpuScalar, 0));

  // HANDLE_CUDA_ERROR(cudaFree(gpuScalar));
  HANDLE_CUBLAS_ERROR(
      cublasZscal(dynamics::Context::getCurrentContext()->getCublasHandle(),
                  dimension, reinterpret_cast<const cuDoubleComplex *>(&scalar),
                  reinterpret_cast<cuDoubleComplex *>(devicePtr), 1));

  return *this;
}

CuDensityMatState cudaq::CuDensityMatState::operator*(double scalar) const {
  CuDensityMatState result = CuDensityMatState::clone(*this);
  result *= scalar;
  return result;
}
} // namespace cudaq
