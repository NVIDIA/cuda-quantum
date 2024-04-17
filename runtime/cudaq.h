/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/NoiseModel.h"
#include "cudaq/host_config.h"
#include "cudaq/qis/qubit_qis.h"
#include <string>
#include <type_traits>

namespace cudaq {
namespace __internal__ {
std::string demangle_kernel(const char *);
bool isLibraryMode(const std::string &);
extern bool globalFalse;
} // namespace __internal__

/// @brief Given a string kernel name, return the corresponding Quake code
std::string get_quake_by_name(const std::string &kernelName);

// Simple test to see if the QuantumKernel template
// type is a `cudaq::builder` with `operator()(Args...)`
template <class T, class = void>
struct hasToQuakeMethod : std::false_type {};
template <class T>
struct hasToQuakeMethod<
    T, typename voider<decltype(std::declval<T>().to_quake())>::type>
    : std::true_type {};

template <class T, class = void>
struct hasCallMethod : std::false_type {};
template <class T>
struct hasCallMethod<
    T, typename voider<decltype(std::declval<T>().operator())>::type>
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
  while (name.size() > 0 && std::isdigit(name[0]))
    name = name.substr(1);
  return name;
}

template <typename Arg, typename... Args>
std::string expand_parameter_pack() {
  if constexpr (sizeof...(Args)) {
    return get_kernel_name_from_type<Arg>() + expand_parameter_pack<Args...>();
  } else {
    return get_kernel_name_from_type<Arg>();
  }
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

/// Get the name of a plain old function that is marked as a quantum kernel.
inline std::string get_kernel_function_name(std::string &&name) {
  return "function_" + std::move(name);
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
#if CUDAQ_USE_STD20
template <typename MemberArg0, typename... MemberArgs, typename QuantumKernel,
          std::enable_if_t<std::is_class_v<std::remove_cvref_t<QuantumKernel>>,
                           bool> = true>
#else
template <typename MemberArg0, typename... MemberArgs, typename QuantumKernel,
          std::enable_if_t<std::is_class_v<std::remove_cv_t<
                               std::remove_reference_t<QuantumKernel>>>,
                           bool> = true>
#endif
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

#if CUDAQ_USE_STD20
template <typename QuantumKernel,
          std::enable_if_t<std::is_class_v<std::remove_cvref_t<QuantumKernel>>,
                           bool> = true>
#else
template <typename QuantumKernel,
          std::enable_if_t<std::is_class_v<std::remove_cv_t<
                               std::remove_reference_t<QuantumKernel>>>,
                           bool> = true>
#endif
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

typedef std::size_t (*KernelArgsCreator)(void **, void **);
KernelArgsCreator getArgsCreator(const std::string &kernelName);

bool kernelHasConditionalFeedback(const std::string &kernelName);

/// @brief Provide a hook to set the target backend.
void set_target_backend(const char *backend);

/// @brief Utility function for setting the shots on the platform
[[deprecated("Specify the number of shots in the using the overloaded sample() "
             "and observe() functions")]] void
set_shots(const std::size_t nShots);

/// @brief Set a custom noise model for simulation. The caller must also call
/// `cudaq::unset_noise` before `model` gets deallocated or goes out of scope.
void set_noise(const cudaq::noise_model &model);

/// @brief Remove an existing noise model from simulation.
void unset_noise();

/// @brief Utility function for clearing the shots
[[deprecated("Specify the number of shots in the using the overloaded sample() "
             "and observe() functions")]] void
clear_shots(const std::size_t nShots);

/// @brief Set a seed for any random number
/// generators used in backend simulations.
void set_random_seed(std::size_t seed);

/// @brief Get a previously set random seed
std::size_t get_random_seed();

/// @brief The number of available GPUs.
int num_available_gpus();

namespace mpi {
/// @brief Return true if CUDA Quantum has MPI plugin support.
bool available();

/// @brief Initialize MPI if available. This function
/// is a no-op if there CUDA Quantum has not been built
/// against MPI.
void initialize();

/// @brief Initialize MPI if available. This function
/// is a no-op if there CUDA Quantum has not been built
/// against MPI. Takes program arguments as input.
void initialize(int argc, char **argv);

/// @brief Return the rank of the calling process.
int rank();

/// @brief Return the number of MPI ranks.
int num_ranks();

/// @brief Return true if MPI is already initialized, false otherwise.
bool is_initialized();

namespace details {
#define CUDAQ_ALL_REDUCE_DEF(TYPE, BINARY)                                     \
  TYPE allReduce(const TYPE &, const BINARY<TYPE> &);

CUDAQ_ALL_REDUCE_DEF(float, std::plus)
CUDAQ_ALL_REDUCE_DEF(float, std::multiplies)

CUDAQ_ALL_REDUCE_DEF(double, std::plus)
CUDAQ_ALL_REDUCE_DEF(double, std::multiplies)

} // namespace details

/// @brief Reduce all values across ranks with the specified binary function.
template <typename T, typename BinaryFunction>
T all_reduce(const T &localValue, const BinaryFunction &function) {
  return details::allReduce(localValue, function);
}

/// @brief Gather all vector data (floating point numbers) locally into the
/// provided global vector.
///
/// Global vector must be sized to fit all vector
/// elements coming from individual ranks.
void all_gather(std::vector<double> &global, const std::vector<double> &local);

/// @brief Gather all vector data (integers) locally into the provided
/// global vector.
///
/// Global vector must be sized to fit all
/// vector elements coming from individual ranks.
void all_gather(std::vector<int> &global, const std::vector<int> &local);

/// @brief Broadcast a vector from a process (rootRank) to all other processes.
void broadcast(std::vector<double> &data, int rootRank);

/// @brief Broadcast a string from a process (rootRank) to all other processes.
void broadcast(std::string &data, int rootRank);

/// @brief Finalize MPI. This function
/// is a no-op if there CUDA Quantum has not been built
/// against MPI.
void finalize();

} // namespace mpi

} // namespace cudaq

// Users should get sample by default
#include "cudaq/algorithms/sample.h"
// Users should get observe by default
#include "cudaq/algorithms/observe.h"
// Users should get get_state by default
#include "cudaq/algorithms/get_state.h"
