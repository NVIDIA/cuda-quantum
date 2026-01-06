/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SimulationState.h"
#include "cudaq.h"
#include "cudaq/utils/cudaq_utils.h"
#include <type_traits>
#include <utility>
#include <vector>

namespace cudaq {
/// @brief Implementation of `SimulationState` for quantum device backends.
/// The state is represented by a quantum kernel.
/// Quantum state contains all the information we need to replicate a
/// call to kernel that created the state.
class QPUState : public cudaq::SimulationState {
protected:
  using ArgDeleter = std::function<void(void *)>;

  std::string kernelName;
  /// @brief  Vector of arguments
  // Note: we create a copy of all arguments except pointers.
  std::vector<void *> args;
  /// @brief Deletion functions for the arguments.
  std::vector<ArgDeleter> deleters;

public:
  template <typename T>
  void addArgument(const T &arg) {
    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
      if constexpr (std::is_copy_constructible_v<
                        std::remove_pointer_t<std::decay_t<T>>>) {
        auto ptr = new std::remove_pointer_t<std::decay_t<T>>(*arg);
        args.push_back(ptr);
        deleters.push_back([](void *ptr) {
          delete static_cast<std::remove_pointer_t<std::decay_t<T>> *>(ptr);
        });
      } else {
        throw std::invalid_argument(
            "Unsupported argument type: only pointers to copy-constructible "
            "types and copy-constructible types are supported.");
      }
    } else if constexpr (std::is_copy_constructible_v<std::decay_t<T>>) {
      auto *ptr = new std::decay_t<T>(arg);
      args.push_back(ptr);
      deleters.push_back(
          [](void *ptr) { delete static_cast<std::decay_t<T> *>(ptr); });
    } else {
      throw std::invalid_argument(
          "Unsupported argument type: only pointers to copy-constructible "
          "types and copy-constructible types are supported.");
    }
  }

  /// @brief Constructor
  template <typename... Args>
  QPUState(std::string &&name, Args &&...args) {
    kernelName = name;
    (addArgument(args), ...);
  }

  QPUState() = default;
  QPUState(const QPUState &other)
      : kernelName(other.kernelName), args(other.args), deleters() {}
  virtual ~QPUState() override;

  /// @brief True if the state has amplitudes or density matrix available.
  virtual bool hasData() const override { return false; }

  /// @brief Helper to retrieve (kernel name, `args` pointers)
  virtual std::optional<std::pair<std::string, std::vector<void *>>>
  getKernelInfo() const override;

  /// @brief Return the number of qubits this state represents.
  std::size_t getNumQubits() const override;

  /// @brief Compute the overlap of this state representation with
  /// the provided `other` state, e.g. `<this | other>`.
  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  /// @brief Return the amplitude of the given computational
  /// basis state.
  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  /// @brief Return the amplitudes of the given list of computational
  /// basis states.
  std::vector<std::complex<double>>
  getAmplitudes(const std::vector<std::vector<int>> &basisState) override;

  /// @brief Return the tensor at the given index. Throws
  /// for an invalid tensor index.
  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override;

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override;

  /// @brief Return the element from the tensor at the
  /// given tensor index and at the given indices.
  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;

  /// @brief Create a new subclass specific SimulationState
  /// from the user provided data set.
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override;

  /// @brief Dump a representation of the state to the
  /// given output stream.
  void dump(std::ostream &os) const override;

  /// @brief Return the floating point precision used by the simulation state.
  precision getPrecision() const override;

  /// @brief Destroy the state representation, frees all associated memory.
  void destroyState() override;

  /// @brief Return true if this `SimulationState` wraps data on the GPU.
  bool isDeviceData() const override;

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  void toHost(std::complex<double> *clientAllocatedData,
              std::size_t numElements) const override;

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  void toHost(std::complex<float> *clientAllocatedData,
              std::size_t numElements) const override;
};
} // namespace cudaq
