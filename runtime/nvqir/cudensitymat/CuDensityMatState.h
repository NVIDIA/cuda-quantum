/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/SimulationState.h"
#include <cudensitymat.h>
#include <unordered_map>

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
  std::size_t batchSize = 1;
  // The dimension of a single state in the batch (used for distributed mode).
  // For non-batched states, this equals dimension.
  // For batched states in distributed mode, dimension < batchSize *
  // singleStateDimension.
  std::size_t singleStateDimension = 0;
  bool borrowedData = false;

public:
  // Create a state with a size and data pointer.
  // Note: the underlying cudm state is not yet initialized as we don't know the
  // dimensions of sub-systems.
  // If `borrowed` is true, the state does not own the device data pointer.
  CuDensityMatState(std::size_t s, void *ptr, bool borrowed = false);

  // Default constructor
  CuDensityMatState() {}

  // Create an initial state of a specific type, e.g., uniform distribution.
  static std::unique_ptr<CuDensityMatState> createInitialState(
      cudensitymatHandle_t handle, cudaq::InitialState initial_state,
      const std::unordered_map<std::size_t, std::int64_t> &dimensions,
      bool createDensityMatrix);

  // Create a batched state
  static std::unique_ptr<CuDensityMatState>
  createBatchedState(cudensitymatHandle_t handle,
                     const std::vector<CuDensityMatState *> initial_states,
                     const std::vector<int64_t> &dimensions,
                     bool createDensityState);

  // Split a batched state into individual states.
  // The caller assumes the ownership of the state pointers, e.g., wrap them
  // under `cudaq::state`.
  static std::vector<CuDensityMatState *>
  splitBatchedState(CuDensityMatState &batchedState);

  // Return the number of qubits
  std::size_t getNumQubits() const override;

  // Compute the overlap with another state
  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  // Retrieve the amplitude of a basis state
  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  // Dump the state to the given output stream
  void dump(std::ostream &os) const override;

  // This state is GPU device data, always return true.
  bool isDeviceData() const override { return true; }

  // Return true if this is an array state
  bool isArrayLike() const override { return false; }

  // Return the precision of the state data elements.
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  // Create the state from external data
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

  // Amplitude accessor
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

  // @brief Create a zero state
  static CuDensityMatState zero_like(const CuDensityMatState &other);
  // Clone a state
  static std::unique_ptr<CuDensityMatState>
  clone(const CuDensityMatState &other);
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

  // Returns the batch size
  std::size_t getBatchSize() const { return batchSize; }

  // Initialize a state with cudensitymat
  void initialize_cudm(cudensitymatHandle_t handleToSet,
                       const std::vector<int64_t> &hilbertSpaceDims,
                       int64_t batchSize);

  /// @brief Accumulation in-place with a coefficient
  void accumulate_inplace(const CuDensityMatState &other,
                          const std::complex<double> &coeff = 1.0);

  /// @brief Accumulation operator
  /// @return Accumulates the summation of two states.
  CuDensityMatState &operator+=(const CuDensityMatState &other);

  /// @brief Scalar multiplication operator
  /// @return The new state after multiplying scalar with the current state.
  CuDensityMatState &operator*=(const std::complex<double> &scalar);

private:
  /// Helper method to transform a state vector states to density matrix states.
  // The caller takes ownership of the returned states.
  static std::vector<CuDensityMatState *> convertStateVecToDensityMatrix(
      const std::vector<CuDensityMatState *> svStates, int64_t dmSize);

  /// Helper to aggregate multiple input states into a distributed batched
  /// state.
  static void
  distributeBatchedStateData(CuDensityMatState &batchedState,
                             const std::vector<CuDensityMatState *> inputStates,
                             int64_t singleStateDimension);
};
/// @endcond
} // namespace cudaq
