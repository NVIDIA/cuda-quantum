/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SimulationState.h"
#include "cudaq.h"
#include "cudaq/utils/cudaq_utils.h"
#include "../utils/registry.h"
#include "../qis/qkernel.h"
#include <vector>
#include <type_traits>

#include <iostream>

namespace details {
  #if CUDAQ_USE_STD20
  template <typename T> struct is_class: std::is_class<std::remove_cvref_t<T>> {};
  #else
  template <typename T> struct is_class: std::is_class<std::remove_cv_t<std::remove_reference_t<T>>> {};
  #endif

  template <typename T>
  inline constexpr bool is_class_v = is_class<T>::value;
  
  template <typename QuantumKernel,
            std::enable_if_t<is_class_v<QuantumKernel>, bool> = true, 
            typename Operator = typename cudaq::qkernel_deduction_guide_helper<
              decltype(&QuantumKernel::operator())>::type>
  cudaq::qkernel<Operator> createQKernel(const QuantumKernel &&kernel) {
    return {&kernel.operator()};
  }

  template <typename QuantumKernel,
            std::enable_if_t<!is_class_v<QuantumKernel>, bool> = true>
  cudaq::qkernel<QuantumKernel> createQKernel(const QuantumKernel &&kernel) {
    return {kernel};
  }
} // namespace details

// /// Helper for creating `cudaq::qkernel` of the correct type.
// template<typename QuantumKernel>
// class QKernelCreator { 
// #if CUDAQ_USE_STD20
//   using type = std::remove_cvref_t<QuantumKernel>;
// #else
//   using type = std::remove_cv_t<std::remove_reference_t<QuantumKernel>>;
// #endif
//   using function_type = std::conditional_t<std::is_class_v<type>, decltype(&type::operator()), type>;

// public:
//   using qkernel_type = cudaq::qkernel<function_type>;
//   qkernel_type qKernel;

// #if CUDAQ_USE_STD20
//   QKernelCreator(const QuantumKernel &&kernel): qKernel{kernel} {}
// #else

//   QKernelCreator(const QuantumKernel &&kernel) {
//     if constexpr (std::is_class_v<type>) {
//       qKernel = qkernel_type{&kernel.operator()};
//     } else {
//       qKernel = qkernel_type{kernel};
//     }
//   }
//   // template<typename = std::enable_if_t<std::is_class_v<type>>>
//   // QKernelCreator(QuantumKernel &&kernel): qKernel{kernel.operator()} {}

//   // template<typename = std::enable_if_t<!std::is_class_v<type>>>
//   // QKernelCreator(QuantumKernel &&kernel): qKernel{kernel} {}
// #endif
// };


namespace cudaq {
/// @brief Implementation of `SimulationState` for quantum device backends.
/// The state is represented by a quantum kernel.
/// Quantum state contains all the information we need to replicate a
/// call to kernel that created the state.
class QPUState : public cudaq::SimulationState {
protected:
  using ArgDeleter = std::function<void(void *)>;
  
  std::string kernelName;
  // void *qKernel = nullptr;
  // ArgDeleter qKernelDeleter;

  /// @brief  Vector of arguments
  // Note: we create a copy of all arguments except pointers.
  std::vector<void *> args;
  /// @brief Deletion functions for the arguments.
  std::vector<ArgDeleter> deleters;

public:
  // template <typename QuantumKernel>
  // void addKernel(const QuantumKernel  &&kernel) {
  //   qKernel = new cudaq::qkernel<QuantumKernel>(kernel);
  //   qKernelDeleter = [](void *ptr) {
  //     delete static_cast<cudaq::qkernel<QuantumKernel>*>(ptr);
  //   };
  // }

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

  // template <typename QuantumKernel, typename... Args>
  // QPUState(const QuantumKernel &&kernel, Args &&...args) {
  //   QKernelCreator<QuantumKernel> creator(kernel);
  //   auto key = cudaq::registry::__cudaq_getLinkableKernelKey(&creator.qKernel);
  //   auto name = cudaq::registry::getLinkableKernelNameOrNull(key);
  //   if (!name)
  //     throw std::runtime_error("Cannot determine kernel name in QPUState");

  //   kernelName = name;
  //   (addArgument(args), ...);
  // }



  /// @brief Constructor
  template <typename QuantumKernel, typename... Args>
  QPUState(const QuantumKernel &&kernel, Args &&...args) {
    auto qKernel = ::details::createQKernel(std::forward<QuantumKernel>(kernel));
    auto key = cudaq::registry::__cudaq_getLinkableKernelKey(&qKernel);
    auto name = cudaq::registry::getLinkableKernelNameOrNull(key);
    if (!name)
      throw std::runtime_error("Cannot determine kernel name in QPUState");

    kernelName = name;
    (addArgument(args), ...);
  }

//   /// @brief Constructor
// #if CUDAQ_USE_STD20
// template <typename QuantumKernel,
//           std::enable_if_t<std::is_class_v<std::remove_cvref_t<QuantumKernel>>, bool> = true, 
//           typename Operator = typename cudaq::qkernel_deduction_guide_helper<
//             decltype(&QuantumKernel::operator())>::type, 
//           typename... Args>
// #else
// template <typename QuantumKernel,
//           std::enable_if_t<std::is_class_v<std::remove_cv_t<std::remove_reference_t<QuantumKernel>>>, bool> = true, 
//           typename Operator = typename cudaq::qkernel_deduction_guide_helper<
//             decltype(&QuantumKernel::operator())>::type, 
//           typename... Args>
// #endif
//   QPUState(const QuantumKernel &&kernel, Args &&...args) {

// #if CUDAQ_USE_STD20
//     cudaq::qkernel<Operator> qKernel{kernel};
// #else
//     cudaq::qkernel<Operator> qKernel{&kernel.operator()};
// #endif
//     auto key = cudaq::registry::__cudaq_getLinkableKernelKey(&qKernel);
//     auto name = cudaq::registry::getLinkableKernelNameOrNull(key);
//     if (!name)
//       throw std::runtime_error("Cannot determine kernel name in QPUState");

//     kernelName = name;
//     (addArgument(args), ...);
//   }

//   /// @brief Constructor
// #if CUDAQ_USE_STD20
// template <typename QuantumKernel,
//           std::enable_if_t<!std::is_class_v<std::remove_cvref_t<QuantumKernel>>, bool> = true, 
//           typename... Args>
// #else
// template <typename QuantumKernel,
//           std::enable_if_t<!std::is_class_v<std::remove_cv_t<std::remove_reference_t<QuantumKernel>>>, bool> = true, 
//           typename... Args>
// #endif
//   QPUState(const QuantumKernel &&kernel, Args &&...args) {
//     cudaq::qkernel<QuantumKernel> qKernel{kernel};
//     auto key = cudaq::registry::__cudaq_getLinkableKernelKey(&qKernel);
//     auto name = cudaq::registry::getLinkableKernelNameOrNull(key);
//     if (!name)
//       throw std::runtime_error("Cannot determine kernel name in QPUState");

//     kernelName = name;
//     (addArgument(args), ...);
//   }

  // /// @brief Constructor
  // template <typename QuantumKernel, typename... Args>
  // QPUState(QuantumKernel &&kernel, Args &&...args) {
  //   if constexpr (has_name<QuantumKernel>::value) {
  //     // kernel_builder kernel: need to JIT code to get it registered.
  //     static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  //     kernelName = kernel.name();
  //   } else {
  //     kernelName = cudaq::getKernelName(kernel);
  //   }
  //   (addArgument(args), ...);
  // }
  QPUState() = default;
  QPUState(const QPUState &other)
      : kernelName(other.kernelName), args(other.args), deleters() {}
  virtual ~QPUState();

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
