/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <vector>

namespace cudaq {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

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

  /// @brief Return the shape of the state data.
  virtual std::vector<std::size_t> getDataShape() const = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided `other` state, e.g. `<this | other>`.
  virtual double overlap(const SimulationState &other) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided host data vector.
  virtual double overlap(const std::vector<complex128> &hostData) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided host data vector.
  virtual double overlap(const std::vector<complex64> &hostData) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided `complex128` data pointer. Clients must provide the number of
  /// elements and the size of each element.
  virtual double overlap(complex128 *deviceOrHostPointer,
                         std::size_t numElements) = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided `complex64` data pointer. Clients must provide the number of
  /// elements and the size of each element.
  virtual double overlap(complex64 *deviceOrHostPointer,
                         std::size_t numElements) = 0;

  /// @brief Dump a representation of the state to the
  /// given output stream.
  virtual void dump(std::ostream &os) const = 0;

  /// @brief Return the internal pointer to the data. This may
  /// be a host or device pointer.
  virtual void *ptr() const = 0;

  // @brief Return the floating point precision used by the simulation state.
  virtual precision getPrecision() const = 0;

  /// @brief Destroy the state representation, frees all associated memory.
  virtual void destroyState() = 0;

  /// @brief Return the vector element at the given index. This method
  /// will throw an exception if the underlying data shape is not
  /// `{N}`.
  virtual complex128 vectorElement(std::size_t idx) {
    throw std::runtime_error("SimulationState::vectorElement not implemented.");
  }

  /// @brief For density-matrix representations, return the element
  /// at the given matrix index. This method will throw an exception
  /// if the underlying data shape is not `{N,M}`.
  virtual complex128 matrixElement(std::size_t i, std::size_t j) {
    throw std::runtime_error("SimulationState::matrixElement not implemented.");
  }

  /// @brief Return the amplitude of the given computational
  /// basis state.
  virtual complex128 getAmplitude(const std::vector<int> &basisState) {
    throw std::runtime_error("SimulationState::getAmplitude not implemented.");
  }

  /// @brief Return the number of elements in this state representation.
  /// Defaults to multiplying all shape elements.
  virtual std::size_t getNumElements() const {
    if (getDataShape().empty())
      return 0;
    std::size_t ret = getDataShape()[0];
    for (std::size_t i = 1; i < getDataShape().size(); i++)
      ret *= getDataShape()[i];
    return ret;
  }

  /// @brief Return true if this `SimulationState` wraps data on the GPU.
  virtual bool isDeviceData() const { return false; }

  /// @brief Return true if this `SimulationState` is vector-like.
  virtual bool isVectorLike() const { return getDataShape().size() == 1; };

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  virtual void toHost(complex128 *clientAllocatedData,
                      std::size_t numElements) const {
    throw std::runtime_error(
        "SimulationState::toHost complex128 not implemented.");
  }

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  virtual void toHost(complex64 *clientAllocatedData,
                      std::size_t numElements) const {
    throw std::runtime_error(
        "SimulationState::toHost complex64 not implemented.");
  }

  /// @brief Destructor
  virtual ~SimulationState() {}
};
} // namespace cudaq