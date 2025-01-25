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
class cudm_mat_state {
public:
  /// @brief To initialize state with raw data.
  explicit cudm_mat_state(std::vector<std::complex<double>> rawData);

  /// @brief Destructor to clean up resources
  ~cudm_mat_state();

  /// @brief Initialize the state as a density matrix or state vector based on
  /// dimensions.
  /// @param hilbertSpaceDims Vector representing the Hilbert Space dimensions.
  void init_state(const std::vector<int64_t> &hilbertSpaceDims);

  /// @brief Check if the state is initialized.
  /// @return True if the state is initialized, false otherwise.
  bool is_initialized() const;

  /// @brief Check if the state is a density matrix.
  /// @return True if the state is a density matrix, false otherwise.
  bool is_density_matrix() const;

  /// @brief Dump the state data to a string for debugging purposes.
  /// @return String representation of the state data.
  std::string dump() const;

  /// @brief Convert the state vector to a density matrix.
  /// @return A new cudm_mat_state representing the density matrix.
  cudm_mat_state to_density_matrix() const;

  /// @brief Get the underlying implementation (if any).
  /// @return The underlying state implementation.
  cudensitymatState_t get_impl() const;

  /// @brief Attach raw data to the internal state representation
  void attach_storage();

private:
  std::vector<std::complex<double>> rawData_;
  cudensitymatState_t state_;
  cudensitymatHandle_t handle_;
  std::vector<int64_t> hilbertSpaceDims_;

  /// @brief Calculate the size of the state vector for the given Hilbert space
  /// dimensions.
  /// @param hilbertSpaceDims Hilbert space dimensions.
  /// @return Size of the state vector.
  size_t calculate_state_vector_size(
      const std::vector<int64_t> &hilbertSpaceDims) const;

  /// @brief Calculate the size of the density matrix for the given Hilbert
  /// space dimensions.
  /// @param hilbertSpaceDims Hilbert space dimensions.
  /// @return Size of the density matrix.
  size_t calculate_density_matrix_size(
      const std::vector<int64_t> &hilbertSpaceDims) const;
};

} // namespace cudaq
