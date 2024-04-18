/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <memory>
#include <variant>
#include <vector>

namespace nvqir {
class TensorNetState;
}

namespace cudaq {
class SimulationState;
class TensorNetworkState {
public:
  virtual std::unique_ptr<nvqir::TensorNetState> reconstructBackendState() = 0;
  virtual std::unique_ptr<cudaq::SimulationState> toSimulationState() = 0;
  virtual ~TensorNetworkState() {}
};

/// @brief state_data is a variant type
/// encoding different forms of user state vector data
/// we support.
using state_data = std::variant<
    std::vector<std::complex<double>>, std::vector<std::complex<float>>,
    std::pair<std::complex<double> *, std::size_t>,
    std::pair<std::complex<float> *, std::size_t>,
    std::vector<std::complex<double> *>, std::shared_ptr<TensorNetworkState>>;

/// @brief The `SimulationState` interface provides and extension point
/// for concrete circuit simulation sub-types to describe their
/// underlying quantum state data in an efficient manner. The `SimulationState`
/// provides a handle on the data for clients, with data remaining on GPU
/// device or CPU memory. The primary goal of this type and its sub-types
/// is to minimize data transfers for the state.
class SimulationState {
protected:
  /// @brief Utility method to inspect the state_data variant
  /// and extract the data pointer and size.
  template <typename T, typename ScalarType = double>
  auto getSizeAndPtrFromVec(const state_data &data) {
    if constexpr (!std::is_same_v<T, ScalarType>)
      throw std::runtime_error("[sim-state] invalid data precision.");

    auto &vec = std::get<std::vector<std::complex<T>>>(data);
    return std::make_tuple(
        vec.size(),
        reinterpret_cast<void *>(const_cast<std::complex<T> *>(vec.data())));
  };

  /// @brief Utility method to inspect the state_data variant
  /// and extract the data pointer and size.
  template <typename T, typename ScalarType = double>
  auto getSizeAndPtrFromPair(const state_data &data) {
    if constexpr (!std::is_same_v<T, ScalarType>)
      throw std::runtime_error("[sim-state] invalid data precision.");

    auto &pair = std::get<std::pair<std::complex<T> *, std::size_t>>(data);
    return std::make_tuple(pair.second, reinterpret_cast<void *>(pair.first));
  };

  /// @brief Utility method to inspect the state_data variant
  /// and extract the data pointer and size.
  template <typename ScalarType = double>
  auto getSizeAndPtr(const state_data &data) {
    auto type = data.index();
    std::tuple<std::size_t, void *> sizeAndPtr;
    if (type == 0)
      sizeAndPtr = getSizeAndPtrFromVec<double, ScalarType>(data);
    else if (type == 1)
      sizeAndPtr = getSizeAndPtrFromVec<float, ScalarType>(data);
    else if (type == 2)
      sizeAndPtr = getSizeAndPtrFromPair<double, ScalarType>(data);
    else if (type == 3)
      sizeAndPtr = getSizeAndPtrFromPair<float, ScalarType>(data);
    else
      throw std::runtime_error("unsupported data type for state.");

    return sizeAndPtr;
  }

  /// @brief Subclass-specific creator method for
  /// new SimulationState instances. Create from the size
  /// and data pointer, which may be on GPU device.
  virtual std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t, void *, std::size_t dataType) = 0;

public:
  /// @brief Runtime-known precision for the simulation data
  enum class precision { fp32, fp64 };

  /// @brief Simulation states are made up of a
  /// vector of Tensors. Each tensor tracks its data pointer
  /// and the tensor extents.
  struct Tensor {
    void *data = nullptr;
    std::vector<std::size_t> extents;
    precision fp_precision;
    std::size_t get_rank() const { return extents.size(); }
    std::size_t get_num_elements() const {
      std::size_t num = 1;
      for (auto &el : extents)
        num *= el;
      return num;
    }
    std::size_t element_size() const {
      return fp_precision == precision::fp32 ? sizeof(std::complex<float>)
                                             : sizeof(std::complex<double>);
    }
  };

  /// @brief Create a new subclass specific SimulationState
  /// from the user provided data set.
  virtual std::unique_ptr<cudaq::SimulationState>
  createFromData(const state_data &data) {
    if (std::holds_alternative<std::shared_ptr<TensorNetworkState>>(data)) {
      auto tensorNetState = std::get<std::shared_ptr<TensorNetworkState>>(data);
      if (tensorNetState)
        return tensorNetState->toSimulationState();
      return nullptr;
    }

    // Flat array state data
    // Check the precision first. Get the size and
    // data pointer from the input data.

    if (getPrecision() == precision::fp32) {
      auto [size, ptr] = getSizeAndPtr<float>(data);
      return createFromSizeAndPtr(size, ptr, data.index());
    }

    auto [size, ptr] = getSizeAndPtr(data);
    return createFromSizeAndPtr(size, ptr, data.index());
  }

  /// @brief Return the tensor at the given index. Throws
  /// for an invalid tensor index.
  virtual Tensor getTensor(std::size_t tensorIdx = 0) const = 0;

  /// @brief Return all tensors that represent this state
  virtual std::vector<Tensor> getTensors() const = 0;

  /// @brief Return the number of tensors that represent this state.
  virtual std::size_t getNumTensors() const = 0;

  /// @brief Return the number of qubits this state represents.
  virtual std::size_t getNumQubits() const = 0;

  /// @brief Compute the overlap of this state representation with
  /// the provided `other` state, e.g. `<this | other>`.
  virtual std::complex<double> overlap(const SimulationState &other) = 0;

  /// @brief Return the amplitude of the given computational
  /// basis state.
  virtual std::complex<double>
  getAmplitude(const std::vector<int> &basisState) = 0;

  /// @brief Dump a representation of the state to the
  /// given output stream.
  virtual void dump(std::ostream &os) const = 0;

  // @brief Return the floating point precision used by the simulation state.
  virtual precision getPrecision() const = 0;

  /// @brief Destroy the state representation, frees all associated memory.
  virtual void destroyState() = 0;

  /// @brief Return the element from the tensor at the
  /// given tensor index and at the given indices.
  virtual std::complex<double>
  operator()(std::size_t tensorIdx, const std::vector<std::size_t> &indices) {
    if (!isArrayLike())
      throw std::runtime_error(
          "Element extraction by linear indexing not supported by this "
          "SimulationState. Please use getAmplitude.");
    throw std::runtime_error("Internal error: Failed to implement linear "
                             "indexing for array-like SimulationState.");
  }

  /// @brief Return the number of elements in this state representation.
  /// Defaults to adding all shape elements.
  virtual std::size_t getNumElements() const {
    std::size_t num = 0;
    for (auto &tensor : getTensors())
      num += tensor.get_num_elements();
    return num;
  }

  /// @brief Return true if this `SimulationState` wraps data on the GPU.
  virtual bool isDeviceData() const { return false; }

  /// @brief Return true if this `SimulationState` wraps contiguous memory
  /// (array-like).
  //  If true, `operator()` can be used to index elements in a multi-dimensional
  //  array manner.
  // Otherwise, `getAmplitude()` must be used.
  virtual bool isArrayLike() const { return true; }

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  virtual void toHost(std::complex<double> *clientAllocatedData,
                      std::size_t numElements) const {
    throw std::runtime_error(
        "SimulationState::toHost complex128 not implemented.");
  }

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  virtual void toHost(std::complex<float> *clientAllocatedData,
                      std::size_t numElements) const {
    throw std::runtime_error(
        "SimulationState::toHost complex64 not implemented.");
  }

  /// @brief Destructor
  virtual ~SimulationState() {}
};
} // namespace cudaq