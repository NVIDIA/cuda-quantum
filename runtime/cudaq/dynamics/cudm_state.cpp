/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <cudaq/cudm_state.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace cudaq {

cudm_state::cudm_state(cudensitymatHandle_t handle,
                       const std::vector<std::complex<double>> rawData,
                       const std::vector<int64_t> &hilbertSpaceDims)
    : rawData_(rawData), state_(nullptr), handle_(handle),
      hilbertSpaceDims_(hilbertSpaceDims) {

  if (rawData_.empty()) {
    throw std::invalid_argument("Raw data cannot be empty.");
  }

  // Allocate device memory
  size_t dataSize = rawData_.size() * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&gpuData_), dataSize));

  // Copy data from host to device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(gpuData_, rawData_.data(), dataSize, cudaMemcpyHostToDevice));

  // Determine if this is a denisty matrix or state vector
  size_t rawDataSize = rawData_.size();
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
      handle_, purity, static_cast<int32_t>(hilbertSpaceDims.size()),
      hilbertSpaceDims.data(), 1, CUDA_C_64F, &state_));

  attach_storage();
}

cudm_state::cudm_state(cudm_state &&other) noexcept
    : rawData_(std::move(other.rawData_)), gpuData_(other.gpuData_),
      state_(other.state_), handle_(other.handle_),
      hilbertSpaceDims_(std::move(other.hilbertSpaceDims_)) {
  other.gpuData_ = nullptr;
  other.state_ = nullptr;
}

cudm_state &cudm_state::operator=(cudm_state &&other) noexcept {
  if (this != &other) {
    // Free existing resources
    if (state_) {
      cudensitymatDestroyState(state_);
    }
    if (gpuData_) {
      cudaFree(gpuData_);
    }

    // Move data from other
    rawData_ = std::move(other.rawData_);
    gpuData_ = other.gpuData_;
    state_ = other.state_;
    handle_ = other.handle_;
    hilbertSpaceDims_ = std::move(other.hilbertSpaceDims_);

    // Nullify other
    other.gpuData_ = nullptr;
    other.state_ = nullptr;
  }

  return *this;
}

cudm_state::~cudm_state() {
  if (state_) {
    cudensitymatDestroyState(state_);
    state_ = nullptr;
  }
  if (gpuData_) {
    cudaFree(gpuData_);
    gpuData_ = nullptr;
  }
}

bool cudm_state::is_initialized() const { return state_ != nullptr; }

bool cudm_state::is_density_matrix() const {
  if (!is_initialized()) {
    return false;
  }

  return rawData_.size() == calculate_density_matrix_size(hilbertSpaceDims_);
}

std::vector<std::complex<double>> cudm_state::get_raw_data() const {
  return rawData_;
}

void *cudm_state::get_device_pointer() const { return gpuData_; }

std::vector<int64_t> cudm_state::get_hilbert_space_dims() const {
  return hilbertSpaceDims_;
}

cudensitymatHandle_t cudm_state::get_handle() const { return handle_; }

cudm_state cudm_state::operator+(const cudm_state &other) const {
  if (rawData_.size() != other.rawData_.size()) {
    throw std::invalid_argument("State size mismatch for addition.");
  }

  cudm_state result = cudm_state(handle_, rawData_, hilbertSpaceDims_);

  double scalingFactor = 1.0;
  double *gpuScalingFactor;
  cudaMalloc(reinterpret_cast<void **>(&gpuScalingFactor), sizeof(double));
  cudaMemcpy(gpuScalingFactor, &scalingFactor, sizeof(double),
             cudaMemcpyHostToDevice);

  HANDLE_CUDM_ERROR(cudensitymatStateComputeAccumulation(
      handle_, other.get_impl(), result.get_impl(), gpuScalingFactor, 0));

  cudaFree(gpuScalingFactor);

  return result;
}

cudm_state &cudm_state::operator+=(const cudm_state &other) {
  if (rawData_.size() != other.rawData_.size()) {
    throw std::invalid_argument("State size mismatch for addition.");
  }

  double scalingFactor = 1.0;
  double *gpuScalingFactor;
  cudaMalloc(reinterpret_cast<void **>(&gpuScalingFactor), sizeof(double));
  cudaMemcpy(gpuScalingFactor, &scalingFactor, sizeof(double),
             cudaMemcpyHostToDevice);

  HANDLE_CUDM_ERROR(cudensitymatStateComputeAccumulation(
      handle_, other.get_impl(), state_, gpuScalingFactor, 0));

  cudaFree(gpuScalingFactor);

  return *this;
}
cudm_state &cudm_state::operator*=(const std::complex<double> &scalar) {
  void *gpuScalar;
  HANDLE_CUDA_ERROR(cudaMalloc(&gpuScalar, sizeof(std::complex<double>)));
  HANDLE_CUDA_ERROR(cudaMemcpy(gpuScalar, &scalar, sizeof(std::complex<double>),
                               cudaMemcpyHostToDevice));

  HANDLE_CUDM_ERROR(
      cudensitymatStateComputeScaling(handle_, state_, gpuScalar, 0));

  HANDLE_CUDA_ERROR(cudaFree(gpuScalar));

  return *this;
}

cudm_state cudm_state::operator*(double scalar) const {
  void *gpuScalar;
  HANDLE_CUDA_ERROR(cudaMalloc(&gpuScalar, sizeof(std::complex<double>)));

  std::complex<double> complexScalar(scalar, 0.0);
  HANDLE_CUDA_ERROR(cudaMemcpy(gpuScalar, &complexScalar,
                               sizeof(std::complex<double>),
                               cudaMemcpyHostToDevice));

  cudm_state result(handle_, rawData_, hilbertSpaceDims_);

  HANDLE_CUDM_ERROR(
      cudensitymatStateComputeScaling(handle_, result.state_, gpuScalar, 0));

  HANDLE_CUDA_ERROR(cudaFree(gpuScalar));

  return result;
}

std::string cudm_state::dump() const {
  if (!is_initialized()) {
    throw std::runtime_error("State is not initialized.");
  }

  std::ostringstream oss;
  oss << "State data: [";
  for (size_t i = 0; i < rawData_.size(); i++) {
    oss << rawData_[i];
    if (i < rawData_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

void cudm_state::dumpDeviceData() const {
  if (!is_initialized()) {
    throw std::runtime_error("State is not initialized.");
  }

  std::vector<std::complex<double>> hostBuffer(rawData_.size());
  HANDLE_CUDA_ERROR(cudaMemcpy(hostBuffer.data(), get_device_pointer(),
                               hostBuffer.size() * sizeof(std::complex<double>),
                               cudaMemcpyDefault));
  std::cout << "State data: [";
  for (size_t i = 0; i < hostBuffer.size(); i++) {
    std::cout << hostBuffer[i];
    if (i < hostBuffer.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

cudm_state cudm_state::to_density_matrix() const {
  if (!is_initialized()) {
    throw std::runtime_error("State is not initialized.");
  }

  if (is_density_matrix()) {
    throw std::runtime_error("State is already a density matrix.");
  }

  std::vector<std::complex<double>> densityMatrix;
  size_t vectorSize = calculate_state_vector_size(hilbertSpaceDims_);
  size_t expectedDensityMatrixSize = vectorSize * vectorSize;
  densityMatrix.resize(expectedDensityMatrixSize);

  for (size_t i = 0; i < vectorSize; i++) {
    for (size_t j = 0; j < vectorSize; j++) {
      densityMatrix[i * vectorSize + j] = rawData_[i] * std::conj(rawData_[j]);
    }
  }

  return cudm_state(handle_, densityMatrix, hilbertSpaceDims_);
}

cudensitymatState_t cudm_state::get_impl() const {
  if (!is_initialized()) {
    throw std::runtime_error("State is not initialized.");
  }
  return state_;
}

void cudm_state::attach_storage() {
  if (!state_) {
    throw std::runtime_error("State is not initialized.");
  }

  if (rawData_.empty() || !gpuData_) {
    throw std::runtime_error("Raw data is empty or device memory not "
                             "allocated. Cannot attach storage.");
  }

  // Retrieve the number of state components
  int32_t numStateComponents;
  HANDLE_CUDM_ERROR(
      cudensitymatStateGetNumComponents(handle_, state_, &numStateComponents));

  // Retrieve the storage size for each component
  std::vector<size_t> componentBufferSizes(numStateComponents);
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(
      handle_, state_, numStateComponents, componentBufferSizes.data()));

  // Validate device memory
  size_t totalSize = std::accumulate(componentBufferSizes.begin(),
                                     componentBufferSizes.end(), 0);
  if (totalSize > rawData_.size() * sizeof(std::complex<double>)) {
    throw std::invalid_argument(
        "Device memory size is insufficient to cover all components.");
  }

  // Attach storage for using device memory (gpuData_)
  std::vector<void *> componentBuffers(numStateComponents);
  size_t offset = 0;
  for (int32_t i = 0; i < numStateComponents; i++) {
    componentBuffers[i] = static_cast<void *>(gpuData_ + offset);
    offset += componentBufferSizes[i] / sizeof(std::complex<double>);
  }

  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
      handle_, state_, numStateComponents, componentBuffers.data(),
      componentBufferSizes.data()));
}

size_t cudm_state::calculate_state_vector_size(
    const std::vector<int64_t> &hilbertSpaceDims) const {
  return std::accumulate(hilbertSpaceDims.begin(), hilbertSpaceDims.end(), 1,
                         std::multiplies<>());
}

size_t cudm_state::calculate_density_matrix_size(
    const std::vector<int64_t> &hilbertSpaceDims) const {
  size_t vectorSize = calculate_state_vector_size(hilbertSpaceDims);
  return vectorSize * vectorSize;
}

// Initialize state based on InitialStateArgT
cudm_state cudm_state::create_initial_state(
    cudensitymatHandle_t handle, const InitialStateArgT &initialStateArg,
    const std::vector<int64_t> &hilbertSpaceDims, bool hasCollapseOps) {
  size_t stateVectorSize =
      std::accumulate(hilbertSpaceDims.begin(), hilbertSpaceDims.end(),
                      static_cast<size_t>(1), std::multiplies<>{});

  std::vector<std::complex<double>> rawData;

  if (std::holds_alternative<InitialState>(initialStateArg)) {
    InitialState initialState = std::get<InitialState>(initialStateArg);

    if (initialState == InitialState::ZERO) {
      rawData.resize(stateVectorSize, {0.0, 0.0});
      // |0> state
      rawData[0] = {1.0, 0.0};
    } else if (initialState == InitialState::UNIFORM) {
      rawData.resize(stateVectorSize, {1.0 / std::sqrt(stateVectorSize), 0.0});
    } else {
      throw std::invalid_argument("Unsupported InitialState type.");
    }
  } else if (std::holds_alternative<void *>(initialStateArg)) {
    void *runtimeState = std::get<void *>(initialStateArg);
    if (!runtimeState) {
      throw std::invalid_argument("Runtime state pointer is null.");
    }

    try {
      auto *externalData =
          reinterpret_cast<std::vector<std::complex<double>> *>(runtimeState);

      if (!externalData || externalData->empty()) {
        throw std::invalid_argument(
            "Runtime state contains invalid or empty data.");
      }

      rawData = *externalData;
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to interpret runtime state: " +
                               std::string(e.what()));
    }
  } else {
    throw std::invalid_argument("Unsupported InitialStateArgT type.");
  }

  cudm_state state(handle, rawData, hilbertSpaceDims);

  // Convert to a density matrix if collapse operators are present.
  if (hasCollapseOps && !state.is_density_matrix()) {
    state = state.to_density_matrix();
  }

  return state;
}
} // namespace cudaq
