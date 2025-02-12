/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cudaq/cudm_error_handling.h>
#include <cudaq/cudm_helpers.h>
#include <cudensitymat.h>
#include <stdexcept>
#include <vector>

namespace cudaq {
// Enum to specify the initial quantum state.
enum class InitialState { ZERO, UNIFORM };

using InitialStateArgT = std::variant<void *, InitialState>;

class cudm_state {
public:
  /// @brief To initialize state with raw data.
  explicit cudm_state(cudensitymatHandle_t handle,
                      const std::vector<std::complex<double>> rawData,
                      const std::vector<int64_t> &hilbertSpaceDims);

  /// @brief To initialize state from a `cudaq::state`
  explicit cudm_state(cudensitymatHandle_t handle, const cudaq::state &simState,
                      const std::vector<int64_t> &hilbertSpaceDims);

  // @brief Create a zero state
  static cudm_state zero_like(const cudm_state &other);

  // Prevent copies (avoids double free issues)
  cudm_state(const cudm_state &) = delete;
  cudm_state &operator=(const cudm_state &) = delete;

  // Allow move semantics
  cudm_state(cudm_state &&other) noexcept;
  cudm_state &operator=(cudm_state &&other) noexcept;

  /// @brief Destructor to clean up resources
  ~cudm_state();

  /// @brief Factory method to create an initial state.
  /// @param InitialStateArgT The type or representation of the initial state.
  /// @param Dimensions of the Hilbert space.
  /// @param hasCollapseOps Whether collapse operators are present.
  /// @return A new 'cudm_state' initialized to the specified state.
  static cudm_state create_initial_state(
      cudensitymatHandle_t handle, const InitialStateArgT &initialStateArg,
      const std::vector<int64_t> &hilbertSpaceDims, bool hasCollapseOps);

  /// @brief Check if the state is initialized.
  /// @return True if the state is initialized, false otherwise.
  bool is_initialized() const;

  /// @brief Check if the state is a density matrix.
  /// @return True if the state is a density matrix, false otherwise.
  bool is_density_matrix() const;

  /// @brief Dump the state data to a string for debugging purposes.
  /// @return String representation of the state data.
  std::string dump() const;

  /// @brief Dump the state data to the console for debugging purposes.
  void dumpDeviceData() const;

  /// @brief Convert the state vector to a density matrix.
  /// @return A new cudm_state representing the density matrix.
  cudm_state to_density_matrix() const;

  /// @brief Get the underlying implementation (if any).
  /// @return The underlying state implementation.
  cudensitymatState_t get_impl() const;

  /// @brief Get a copy of the raw data representing the quantum state.
  /// @return A copy of the raw data as a vector of complex numbers.
  std::vector<std::complex<double>> get_raw_data() const;

  /// @brief Get the pointer to device memory buffer storing the state.
  /// @return GPU device pointer
  void *get_device_pointer() const;

  /// @brief Get a copy of the hilbert space dimensions for the quantum state.
  /// @return A copy of the hilbert space dimensions of a vector of integers.
  std::vector<int64_t> get_hilbert_space_dims() const;

  /// @brief Returns the handle
  /// @return The handle associated with the state
  cudensitymatHandle_t get_handle() const;

  /// @brief Addition operator (element-wise)
  /// @return The new state after the summation of two states.
  cudm_state operator+(const cudm_state &other) const;

  /// @brief Accumulation operator
  /// @return Accumulates the summation of two states.
  cudm_state &operator+=(const cudm_state &other);

  /// @brief Scalar multiplication operator
  /// @return The new state after multiplying scalar with the current state.
  cudm_state &operator*=(const std::complex<double> &scalar);

  cudm_state operator*(double scalar) const;

private:
  // TODO: remove this host raw data, we shouldn't keep this as it will be
  // decoupled to the GPU data.
  std::vector<std::complex<double>> rawData_;
  int64_t gpuDataSize_ = 0;
  std::complex<double> *gpuData_;
  cudensitymatState_t state_;
  cudensitymatHandle_t handle_;
  std::vector<int64_t> hilbertSpaceDims_;
  // Private default constructor
  cudm_state() = default;
  /// @brief Attach raw data storage to GPU
  void attach_storage();

  /// @brief Calculate the size of the state vector for the given Hilbert space
  /// dimensions.
  /// @param hilbertSpaceDims Hilbert space dimensions.
  /// @return Size of the state vector.
  static size_t calculate_state_vector_size(
      const std::vector<int64_t> &hilbertSpaceDims);

  /// @brief Calculate the size of the density matrix for the given Hilbert
  /// space dimensions.
  /// @param hilbertSpaceDims Hilbert space dimensions.
  /// @return Size of the density matrix.
  static size_t
  calculate_density_matrix_size(const std::vector<int64_t> &hilbertSpaceDims);
};

} // namespace cudaq
