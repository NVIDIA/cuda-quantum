/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>

namespace cudaq {
using complex = std::complex<double>;

/// @brief The `SimulationState` interface provides and extension point
/// for concrete circuit simulation sub-types to describe their
/// underlying quantum state data in an efficient manner. The `SimulationState`
/// provides a handle on the data for clients, with data remaining on GPU
/// device or CPU memory. The primary goal of this type and its sub-types
/// is to minimize data transfers for the state.
class SimulationState {
public:
  enum class precision { fp32, fp64 };

  /// @brief Return the number of qubits this state represents.
  virtual std::size_t getNumQubits() const = 0;

  virtual std::size_t getNumElements() const {
    if (getDataShape().empty())
      return 0;
    std::size_t ret = getDataShape()[0];
    for (std::size_t i = 1; i < getDataShape().size(); i++)
      ret *= getDataShape()[i];
    return ret;
  }

  /// @brief Return the shape of the state data.
  virtual std::vector<std::size_t> getDataShape() const = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided `other` state, e.g. `<this | other>`.
  virtual double overlap(const SimulationState &other) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided host data vector.
  virtual double overlap(const std::vector<complex> &hostData) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided host data vector.
  virtual double overlap(const std::vector<std::complex<float>> &hostData) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided data pointer.
  virtual double overlap(void *deviceOrHostPointer) = 0;

  /// @brief Return the vector element at the given index. This method
  /// will throw an exception if the underlying data shape is not
  /// `{N}`.
  virtual complex vectorElement(std::size_t idx) {
    throw std::runtime_error("SimulationState::vectorElement not implemented.");
  }

  /// @brief For density-matrix representations, return the element
  /// at the given matrix index. This method will throw an exception
  /// if the underlying data shape is not `{N,M}`.
  virtual complex matrixElement(std::size_t i, std::size_t j) {
    throw std::runtime_error("SimulationState::matrixElement not implemented.");
  }

  /// @brief Return the amplitude of the given computational
  /// basis state.
  virtual complex getAmplitude(const std::vector<int> &basisState) {
    throw std::runtime_error("SimulationState::getAmplitude not implemented.");
  }

  /// @brief Dump a representation of the state to the
  /// given output stream.
  virtual void dump(std::ostream &os) const = 0;

  /// @brief Return the internal pointer to the data. This may
  /// be a host or device pointer.
  virtual void *ptr() const = 0;

  /// @brief Return true if this `SimulationState` wraps data on the GPU.
  virtual bool isDeviceData() const { return false; }

  /// @brief Return true if this `SimulationState` is vector-like.
  virtual bool isVectorLike() const { return getDataShape().size() == 1; };

  // @brief Return the floating point precision used by the simulation state.
  virtual precision getPrecision() const = 0;

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client.
  virtual void toHost(void *clientAllocatedData) const {
    throw std::runtime_error("SimulationState::toHost not implemented.");
  }

  /// @brief Destroy the state representation, frees all associated memory.
  virtual void destroyState() = 0;

  /// @brief Destructor
  virtual ~SimulationState() {}
};
} // namespace cudaq