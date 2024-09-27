/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/EigenDense.h"
#include "common/Logger.h"
#include "common/SimulationState.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cuda_runtime_api.h>

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(fmt::format("[cuSuperOp] %{} in {} (line {})",  \
                                           cudaGetErrorString(err),            \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  };

namespace cudaq {
class CuSuperOpState : public cudaq::SimulationState {
private:
  bool isDensityMatrix = false;
  std::size_t dimension = 0;
  /// @brief State device data pointer.
  void *devicePtr = nullptr;

public:
  CuSuperOpState(std::size_t s, void *ptr, bool isDm)
      : isDensityMatrix(isDm), devicePtr(ptr),
        dimension(isDm ? std::sqrt(s) : s) {}

  CuSuperOpState() {}

  std::size_t getNumQubits() const override { return std::log2(dimension); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (getTensor().extents != other.getTensor().extents)
      throw std::runtime_error("[CuSuperOpState] overlap error - other state "
                               "dimension not equal to this state dimension.");

    if (other.getPrecision() != getPrecision()) {
      throw std::runtime_error(
          "[CuSuperOpState] overlap error - precision mismatch.");
    }

    if (!isDensityMatrix) {
      // FIXME: to do
      throw std::runtime_error(
          "[CuSuperOpState] overlap error - not implemented.");
      return 0.0;
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
  getAmplitude(const std::vector<int> &basisState) override {
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    const auto extractValue = [&](std::size_t idx) {
      std::complex<double> value;
      HANDLE_CUDA_ERROR(cudaMemcpy(
          &value, reinterpret_cast<std::complex<double> *>(devicePtr) + idx,
          sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
      return value;
    };

    return extractValue(idx);
  }

  /// @brief Dump the state to the given output stream
  void dump(std::ostream &os) const override {
    // get state data from device to print
    Eigen::MatrixXcd state(dimension, isDensityMatrix ? dimension : 1);
    const auto size = isDensityMatrix ? dimension * dimension : dimension;
    HANDLE_CUDA_ERROR(cudaMemcpy(state.data(), devicePtr,
                                 size * sizeof(std::complex<double>),
                                 cudaMemcpyDeviceToHost));
    os << state << std::endl;
  }

  /// @brief This state is GPU device data, always return true.
  bool isDeviceData() const override { return true; }

  bool isArrayLike() const override { return false; }

  /// @brief Return the precision of the state data elements.
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *dataPtr,
                       std::size_t type) override {
    bool isDm = false;
    if (type == cudaq::detail::variant_index<cudaq::state_data,
                                             cudaq::TensorStateData>()) {
      if (size != 1)
        throw std::runtime_error(
            "[CuSuperOpState]: createFromSizeAndPtr expects a single tensor");
      auto *casted =
          reinterpret_cast<cudaq::TensorStateData::value_type *>(dataPtr);

      auto [ptr, extents] = casted[0];
      if (extents.size() > 2)
        throw std::runtime_error("[CuSuperOpState]: createFromSizeAndPtr only "
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
    // printf("Created CuSuperOpState ptr %p\n", devicePtr);
    return std::make_unique<CuSuperOpState>(size, devicePtr, isDm);
  }

  /// @brief Return the tensor at the given index. Throws
  /// for an invalid tensor index.
  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0) {
      throw std::runtime_error(
          "CuSuperOpState state only supports a single tensor");
    }
    const std::vector<std::size_t> extents =
        isDensityMatrix ? std::vector<std::size_t>{dimension, dimension}
                        : std::vector<std::size_t>{dimension};
    return Tensor{devicePtr, extents, precision::fp64};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    const auto extractValue = [&](std::size_t idx) {
      std::complex<double> value;
      HANDLE_CUDA_ERROR(cudaMemcpy(
          &value, reinterpret_cast<std::complex<double> *>(devicePtr) + idx,
          sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
      return value;
    };

    if (tensorIdx != 0)
      throw std::runtime_error(
          "CuSuperOpState state only supports a single tensor");
    if (isDensityMatrix) {
      if (indices.size() != 2)
        throw std::runtime_error("CuSuperOpState holding a density matrix "
                                 "supports only 2-dimensional indices");
      if (indices[0] >= dimension || indices[1] >= dimension)
        throw std::runtime_error("CuSuperOpState indices out of range");
      return extractValue(indices[0] * dimension + indices[1]);
    }
    if (indices.size() != 1)
      throw std::runtime_error("CuSuperOpState holding a state vector supports "
                               "only 1-dimensional indices");
    if (indices[0] >= dimension)
      throw std::runtime_error("Index out of bounds");
    return extractValue(indices[0]);
  }

  /// @brief Copy the state device data to the user-provided host data pointer.
  void toHost(std::complex<double> *userData,
              std::size_t numElements) const override {
    if (numElements != dimension * (isDensityMatrix ? dimension : 1)) {
      throw std::runtime_error("Number of elements in user data does not match "
                               "the size of the state");
    }
    HANDLE_CUDA_ERROR(cudaMemcpy(userData, devicePtr,
                                 numElements * sizeof(std::complex<double>),
                                 cudaMemcpyDeviceToHost));
  }

  /// @brief Copy the state device data to the user-provided host data pointer.
  void toHost(std::complex<float> *userData,
              std::size_t numElements) const override {
    throw std::runtime_error("CuSuperOpState: Data type mismatches - expecting "
                             "double-precision array.");
  }

  /// @brief Free the device data.
  void destroyState() override {
    if (devicePtr != nullptr) {
      HANDLE_CUDA_ERROR(cudaFree(devicePtr));
      devicePtr = nullptr;
      dimension = 0;
      isDensityMatrix = false;
    }
  }
};

} // namespace cudaq