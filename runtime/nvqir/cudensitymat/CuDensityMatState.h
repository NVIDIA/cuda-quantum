/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/SimulationState.h"
#include <cudensitymat.h>

namespace cudaq {
/// @cond
// This is an internal class, no API documentation.
// Simulation state implementation for `CuDensityMatState`.
class CuDensityMatState : public cudaq::SimulationState {
private:
  bool isDensityMatrix = false;
  std::size_t dimension = 0;
  // State device data pointer.
  void *devicePtr = nullptr;

  cudensitymatState_t cudmState = nullptr;
  cudensitymatHandle_t cudmHandle = nullptr;
  std::vector<int64_t> hilbertSpaceDims;

public:
  CuDensityMatState(std::size_t s, void *ptr, bool isDm)
      : isDensityMatrix(isDm), devicePtr(ptr),
        dimension(isDm ? std::sqrt(s) : s) {}

  CuDensityMatState() {}

  std::size_t getNumQubits() const override { return std::log2(dimension); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  // Dump the state to the given output stream
  void dump(std::ostream &os) const override;

  // This state is GPU device data, always return true.
  bool isDeviceData() const override { return true; }

  bool isArrayLike() const override { return false; }

  // Return the precision of the state data elements.
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *dataPtr,
                       std::size_t type) override;

  // Return the tensor at the given index. Throws
  // for an invalid tensor index.
  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  // Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  // Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;

  // Copy the state device data to the user-provided host data pointer.
  void toHost(std::complex<double> *userData,
              std::size_t numElements) const override;

  // Copy the state device data to the user-provided host data pointer.
  void toHost(std::complex<float> *userData,
              std::size_t numElements) const override;
  // Free the device data.
  void destroyState() override;

  // TODO: Tidy this up, remove unnecessary methods
  /// @brief To initialize state with raw data.
  explicit CuDensityMatState(cudensitymatHandle_t handle,
                             const std::vector<std::complex<double>> &rawData,
                             const std::vector<int64_t> &hilbertSpaceDims);
  /// @brief To initialize state from a `cudaq::state`
  explicit CuDensityMatState(cudensitymatHandle_t handle,
                             const CuDensityMatState &simState,
                             const std::vector<int64_t> &hilbertSpaceDims);
  // @brief Create a zero state
  static CuDensityMatState zero_like(const CuDensityMatState &other);
  static CuDensityMatState clone(const CuDensityMatState &other);
  // Prevent copies (avoids double free issues)
  CuDensityMatState(const CuDensityMatState &) = delete;
  CuDensityMatState &operator=(const CuDensityMatState &) = delete;

  // Allow move semantics
  CuDensityMatState(CuDensityMatState &&other) noexcept;
  CuDensityMatState &operator=(CuDensityMatState &&other) noexcept;

  /// @brief Destructor to clean up resources
  ~CuDensityMatState();

  /// @brief Check if the state is initialized.
  /// @return True if the state is initialized, false otherwise.
  bool is_initialized() const;

  /// @brief Check if the state is a density matrix.
  /// @return True if the state is a density matrix, false otherwise.
  bool is_density_matrix() const;

  /// @brief Convert the state vector to a density matrix.
  /// @return A new CuDensityMatState representing the density matrix.
  CuDensityMatState to_density_matrix() const;

  /// @brief Get the underlying implementation (if any).
  /// @return The underlying state implementation.
  cudensitymatState_t get_impl() const;

  /// @brief Get the pointer to device memory buffer storing the state.
  /// @return GPU device pointer
  void *get_device_pointer() const;

  /// @brief Get a copy of the `hilbert` space dimensions for the quantum state.
  /// @return A copy of the `hilbert` space dimensions of a vector of integers.
  std::vector<int64_t> get_hilbert_space_dims() const;

  /// @brief Returns the handle
  /// @return The handle associated with the state
  cudensitymatHandle_t get_handle() const;

  void initialize_cudm(cudensitymatHandle_t handleToSet,
                       const std::vector<int64_t> &hilbertSpaceDims);
  /// @brief Addition operator (element-wise)
  /// @return The new state after the summation of two states.
  CuDensityMatState operator+(const CuDensityMatState &other) const;

  /// @brief Accumulation operator
  /// @return Accumulates the summation of two states.
  CuDensityMatState &operator+=(const CuDensityMatState &other);

  /// @brief Scalar multiplication operator
  /// @return The new state after multiplying scalar with the current state.
  CuDensityMatState &operator*=(const std::complex<double> &scalar);

  CuDensityMatState operator*(double scalar) const;
};
/// @endcond
} // namespace cudaq
