/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/DeviceCodeRegistry.h"
#include "common/NoiseModel.h"
#include "cudaq/host_config.h"
#include "cudaq/qis/qubit_qis.h"
#include <string>
#include <tuple>
#include <type_traits>

namespace cudaq {
namespace details {
// Test std::tuple layout.
constexpr bool isTupleRecursivelyDefined() {
  std::tuple<double, int, char> t;
  return static_cast<void *>(&std::get<double>(t)) != static_cast<void *>(&t);
}
[[maybe_unused]] static bool TupleIsReverse = isTupleRecursivelyDefined();
} // namespace details

namespace __internal__ {
std::string demangle_kernel(const char *);
extern bool globalFalse;
} // namespace __internal__

// Simple test to see if the QuantumKernel template
// type is a `cudaq::builder` with `operator()(Args...)`
template <class T, class = void>
struct hasToQuakeMethod : std::false_type {};
template <class T>
struct hasToQuakeMethod<T, std::void_t<decltype(std::declval<T>().to_quake())>>
    : std::true_type {};

template <class T, class = void>
struct hasCallMethod : std::false_type {};
template <class T>
struct hasCallMethod<
    T, typename std::void_t<decltype(std::declval<T>().operator())>>
    : std::true_type {};

namespace internal {
/// @brief Define some template types to inspect
/// the argument structure of the input QuantumKernel type
template <typename T>
struct KernelCallArgs;

/// @brief Deduce the `Args...` for `operator()(Args...)`
template <typename RT, typename Owner, typename... Args>
struct KernelCallArgs<RT (Owner::*)(Args...)> {
  static constexpr std::size_t ArgCount = sizeof...(Args);
  using ReturnType = RT;
  using ArgsTuple = std::tuple<std::remove_reference_t<Args>...>;
};

/// @brief Deduce the `Args...` for `operator()(Args...)`, for constant lambda
template <typename RT, typename Owner, typename... Args>
struct KernelCallArgs<RT (Owner::*)(Args...) const> {
  static constexpr std::size_t ArgCount = sizeof...(Args);
  using ReturnType = RT;
  using ArgsTuple = std::tuple<std::remove_reference_t<Args>...>;
};

template <typename QuantumKernel>
std::string get_kernel_name_from_type() {
  std::string name = typeid(QuantumKernel).name();
  name.erase(0, name.find_first_not_of("0123456789"));
  return name;
}

template <typename Arg, typename... Args>
std::string expand_parameter_pack() {
  return (get_kernel_name_from_type<Arg>() + ... +
          get_kernel_name_from_type<Args>());
}
} // namespace internal

/// The typical case. The kernel is a C++ callable class, QuantumKernel. Use
/// this when the kernel is a template class, lambda, or plain old class.
template <typename QuantumKernel>
std::string get_kernel_name() {
  return internal::get_kernel_name_from_type<QuantumKernel>();
}

/// Get the name of the kernel when the kernel has a template `operator()`
/// function. The resolved template arguments must be provided as `Args`.
template <typename QuantumKernel, typename... Args>
std::string get_kernel_template_member_name() {
  return "instance_" + internal::get_kernel_name_from_type<QuantumKernel>() +
         internal::expand_parameter_pack<Args...>();
}

inline std::string get_kernel_function_name(const std::string &name) {
  return "function_" + name;
}

/// Get the name of a template function (not a class member) that is marked as a
/// quantum kernel. The template arguments must be supplied as `Args`.
template <typename... Args>
std::string get_kernel_template_function_name(std::string &&funcName) {
  std::string name = internal::expand_parameter_pack<Args...>();
  return "instance_function_" + std::move(funcName) + name;
}

template <typename... Args>
std::string get_kernel_template_function_name(const std::string &funcName) {
  std::string name = internal::expand_parameter_pack<Args...>();
  return "instance_function_" + funcName + name;
}

/// These get_quake overloads can be used for introspection, to look up the
/// Quake IR for a specific kernel by providing an instance of the kernel, etc.
template <typename MemberArg0, typename... MemberArgs, typename QuantumKernel,
          std::enable_if_t<std::is_class_v<std::remove_cvref_t<QuantumKernel>>,
                           bool> = true>
std::string get_quake(QuantumKernel &&kernel) {
  // See comment below.
  if (__internal__::globalFalse) {
    using ArgsTuple = typename internal::KernelCallArgs<
        decltype(&std::remove_reference_t<QuantumKernel>::template
                 operator()<MemberArg0, MemberArgs...>)>::ArgsTuple;
    ArgsTuple args;
    std::apply([&kernel](auto &&...el) { kernel(el...); }, args);
  }
  return get_quake_by_name(
      get_kernel_template_member_name<QuantumKernel, MemberArg0,
                                      MemberArgs...>());
}

template <typename QuantumKernel,
          std::enable_if_t<std::is_class_v<std::remove_cvref_t<QuantumKernel>>,
                           bool> = true>
std::string get_quake(QuantumKernel &&kernel) {
  if constexpr (hasToQuakeMethod<QuantumKernel>::value) {
    return kernel.to_quake();
  } else {
    // If we have template class kernel specializations, we want to ensure they
    // are not optimized away. This code will force the compiler to retain the
    // template class instantiation. The globalFalse flag, ensures this code
    // does not execute. However the compiler will respect the template
    // specialization and not optimize it away.
    if (__internal__::globalFalse) {
      using ArgsTuple = typename internal::KernelCallArgs<
          decltype(&std::remove_reference_t<QuantumKernel>::operator())>::
          ArgsTuple;
      ArgsTuple args;
      std::apply([&kernel](auto &&...el) { kernel(el...); }, args);
    }
    return get_quake_by_name(get_kernel_name<QuantumKernel>());
  }
}

template <typename MemberArg0, typename... MemberArgs>
std::string get_quake(std::string &&functionName) {
  return get_quake_by_name(
      get_kernel_template_function_name<MemberArg0, MemberArgs...>(
          std::move(functionName)));
}

inline std::string get_quake(std::string &&functionName) {
  return get_quake_by_name(get_kernel_function_name(std::move(functionName)));
}

/// @brief Set a custom noise model for simulation. The caller must also call
/// `cudaq::unset_noise` before `model` gets deallocated or goes out of scope.
void set_noise(const cudaq::noise_model &model);

/// @brief Remove an existing noise model from simulation.
void unset_noise();

/// @brief Set a seed for any random number
/// generators used in backend simulations.
void set_random_seed(std::size_t seed);

/// @brief Get a previously set random seed
std::size_t get_random_seed();

/// @brief The number of available GPUs.
int num_available_gpus();

} // namespace cudaq

#include "cudaq/cudaq_mpi.h"

// Users should get sample by default
#include "cudaq/algorithms/sample.h"
// Users should get run by default
#include "cudaq/algorithms/run.h"
// Users should get observe by default
#include "cudaq/algorithms/observe.h"
// Users should get get_state by default
#include "cudaq/algorithms/get_state.h"
// Users should get device.h by default
#include "cudaq/driver/device.h"
// Users should get apply_noise by default
#include "cudaq/apply_noise.h"
