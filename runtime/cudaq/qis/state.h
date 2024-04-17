/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "common/SimulationState.h"

namespace cudaq {

using tensor = SimulationState::Tensor;
/// @brief The cudaq::state encapsulate backend simulation state vector or
/// density matrix data.
class state {

private:
  /// @brief Reference to the simulation data
  std::shared_ptr<SimulationState> internal;

public:
  /// @brief The constructor, takes the simulation data and owns it
  state(SimulationState *ptrToOwn)
      : internal(std::shared_ptr<SimulationState>(ptrToOwn)) {}

  /// @brief Convenience function for extracting from a known vector.
  std::complex<double> operator[](std::size_t idx);

  /// @brief Convenience function for extracting from a known matrix.
  std::complex<double> operator()(std::size_t idx, std::size_t jdx);

  /// @brief General extraction operator for state data.
  std::complex<double> operator()(const std::initializer_list<std::size_t> &,
                                  std::size_t tensorIdx = 0);

  /// @brief Return the tensor at the given index for this state representation.
  /// For state-vector and density matrix simulation states, there is just one
  /// tensor with rank 1 or 2 respectively.
  tensor get_tensor(std::size_t tensorIdx = 0) const;

  /// @brief Return all tensors that represent this simulation state.
  std::vector<tensor> get_tensors() const;

  /// @brief Return the number of tensors that represent this state.
  std::size_t get_num_tensors() const;

  /// @brief Return the underlying floating point precision for
  /// this state.
  SimulationState::precision get_precision() const;

  /// @brief Return true if this a state on the GPU.
  bool is_on_gpu() const;

  /// @brief Copy this state from device to
  template <typename ScalarType>
  void to_host(std::complex<ScalarType> *hostPtr,
               std::size_t numElements) const {
    if (!is_on_gpu())
      throw std::runtime_error("to_host requested, but the state is already on "
                               "host. Check with is_on_gpu() method.");
    internal->toHost(hostPtr, numElements);
  }

  /// @brief Dump the state to standard out
  void dump();

  /// @brief Dump the state to given output stream
  void dump(std::ostream &os);

  /// @brief Compute the overlap of this state
  /// with the other one.
  std::complex<double> overlap(const state &other);

  /// @brief Return the amplitude of the given computational
  /// basis state
  std::complex<double> amplitude(const std::vector<int> &basisState);

  /// @brief Create a new state from user-provided data.
  /// The data can be host or device data.
  static state from_data(const state_data &data);

  ~state();
};

} // namespace cudaq