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

namespace cudaq {

cudm_state::cudm_state(cudensitymatHandle_t handle,
                       std::vector<std::complex<double>> rawData)
    : rawData_(rawData), state_(nullptr), handle_(handle), hilbertSpaceDims_() {
  // Allocate device memory
  size_t dataSize = rawData_.size() * sizeof(std::complex<double>);
  cudaMalloc(reinterpret_cast<void **>(&gpuData_), dataSize);

  // Copy data from host to device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(gpuData_, rawData_.data(), dataSize, cudaMemcpyHostToDevice));
}

cudm_state::~cudm_state() {
  if (state_) {
    cudensitymatDestroyState(state_);
  }
  if (gpuData_) {
    cudaFree(gpuData_);
  }
}

void cudm_state::init_state(const std::vector<int64_t> &hilbertSpaceDims) {
  if (state_) {
    throw std::runtime_error("State is already initialized.");
  }

  hilbertSpaceDims_ = hilbertSpaceDims;

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

std::vector<int64_t> cudm_state::get_hilbert_space_dims() const {
  return hilbertSpaceDims_;
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

  cudm_state densityMatrixState(handle_, densityMatrix);
  densityMatrixState.init_state(hilbertSpaceDims_);
  return densityMatrixState;
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
  size_t size = 1;
  for (auto dim : hilbertSpaceDims) {
    size *= dim;
  }
  return size;
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

  cudm_state state(handle, rawData);
  state.init_state(hilbertSpaceDims);

  // Convert to a density matrix if collapse operators are present.
  if (hasCollapseOps && !state.is_density_matrix()) {
    state = state.to_density_matrix();
  }

  return state;
}
} // namespace cudaq
