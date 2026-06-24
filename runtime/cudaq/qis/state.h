/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SimulationState.h"
#include "cudaq/host_config.h"

namespace cudaq {

class state_helper;

using tensor = SimulationState::Tensor;

/// @brief The cudaq::state encapsulate backend simulation state vector or
/// density matrix data.
class state {

private:
  /// @brief Reference to the simulation data
  std::shared_ptr<SimulationState> internal;
  template <std::size_t>
  friend class qvector;
  template <std::size_t>
  friend class qudit;
  friend class state_helper;

public:
  /// @brief The constructor, takes the simulation data and owns it
  explicit state(SimulationState *ptrToOwn);
  /// @brief Copy constructor (default)
  state(const state &other) = default;

  /// Overloaded constructors.
  /// These construct a `state` from a raw input state vector. The number of
  /// qubits is determined by the size of the input vector. The user is
  /// responsible for providing (and verifying) the element values. These values
  /// must be correct for the simulator that is in use.
  state(const std::vector<std::complex<double>> &vector) { initialize(vector); }
  state(std::vector<std::complex<double>> &&vector) {
    std::vector<std::complex<double>> v{std::move(vector)};
    initialize(v);
  }
  state(const std::vector<std::complex<float>> &vector) { initialize(vector); }
  state(std::vector<std::complex<float>> &&vector) {
    std::vector<std::complex<float>> v{std::move(vector)};
    initialize(v);
  }
  state(const std::vector<double> &vector) {
    initialize(std::vector<cudaq::complex>{vector.begin(), vector.end()});
  }
  state(std::vector<double> &&vector) {
    std::vector<double> v{std::move(vector)};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(const std::vector<float> &vector) {
    initialize(std::vector<cudaq::complex>{vector.begin(), vector.end()});
  }
  state(std::vector<float> &&vector) {
    std::vector<float> v{std::move(vector)};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(const std::initializer_list<std::complex<double>> &list) {
    std::vector<std::complex<double>> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(std::initializer_list<std::complex<double>> &&list) {
    std::vector<std::complex<double>> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(const std::initializer_list<std::complex<float>> &list) {
    std::vector<std::complex<float>> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(std::initializer_list<std::complex<float>> &&list) {
    std::vector<std::complex<float>> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(const std::initializer_list<double> &list) {
    std::vector<double> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(std::initializer_list<double> &&list) {
    std::vector<double> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(const std::initializer_list<float> &list) {
    std::vector<float> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }
  state(std::initializer_list<float> &&list) {
    std::vector<float> v{list.begin(), list.end()};
    initialize(std::vector<cudaq::complex>{v.begin(), v.end()});
  }

  /// @brief Copy assignment
  state &operator=(state &&other);
  /// @brief Default destructor
  ~state() = default;

  /// @brief Convenience function for extracting from a known vector.
  std::complex<double> operator[](std::size_t idx) const;

  /// @brief Convenience function for extracting from a known matrix.
  std::complex<double> operator()(std::size_t idx, std::size_t jdx) const;

  /// @brief General extraction operator for state data.
  std::complex<double> operator()(const std::initializer_list<std::size_t> &,
                                  std::size_t tensorIdx = 0) const;

  /// @brief Return the tensor at the given index for this state representation.
  /// For state-vector and density matrix simulation states, there is just one
  /// tensor with rank 1 or 2 respectively.
  tensor get_tensor(std::size_t tensorIdx = 0) const;

  /// @brief Return all tensors that represent this simulation state.
  std::vector<tensor> get_tensors() const;

  /// @brief Return the number of tensors that represent this state.
  std::size_t get_num_tensors() const;

  /// @brief Return the number of qubits.
  std::size_t get_num_qubits() const;

  /// @brief Return the underlying floating point precision for this state.
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
  void dump() const;

  /// @brief Dump the state to given output stream
  void dump(std::ostream &os) const;

  /// @brief Compute the overlap of this state with the other one.
  /// For state vectors (pure states), it is computed as `|<this | other>|`.
  std::complex<double> overlap(const state &other);

  /// @brief Return the amplitude of the given computational basis state
  std::complex<double> amplitude(const std::vector<int> &basisState);

  /// @brief Return the amplitudes of the given list of computational basis
  /// states
  std::vector<std::complex<double>>
  amplitudes(const std::vector<std::vector<int>> &basisStates);

  /// @brief Create a new state from user-provided data.
  /// The data can be host or device data.
  static state from_data(const state_data &data) {
    return state{}.initialize(data);
  }

private:
  state() : internal{nullptr} {}
  state &initialize(const state_data &data);
};

class state_helper {
public:
  static SimulationState *getSimulationState(cudaq::state *state) {
    return state->internal.get();
  }
};

} // namespace cudaq
