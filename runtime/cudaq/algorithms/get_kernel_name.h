/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/concepts.h"
#include "cudaq/qis/qkernel.h"
#include "cudaq/utils/registry.h"

namespace cudaq {

namespace details {

#ifdef CUDAQ_LIBRARY_MODE
template <typename QuantumKernel>
QuantumKernel createQKernel(QuantumKernel &&kernel) {
  return kernel;
}
#else

// For class types (lambdas, functors)
template <typename QuantumKernel, typename Q = std::decay_t<QuantumKernel>>
std::enable_if_t<std::is_class_v<Q>,
                 cudaq::qkernel<typename cudaq::qkernel_deduction_guide_helper<
                     decltype(&Q::operator())>::type>>
createQKernel(QuantumKernel &&kernel) {
  return {kernel};
}

// For function references - just return the kernel itself
template <typename QuantumKernel>
std::enable_if_t<std::is_function_v<std::remove_reference_t<QuantumKernel>>,
                 QuantumKernel>
createQKernel(QuantumKernel &&kernel) {
  return kernel;
}

// For function pointers - just return the kernel itself
template <typename QuantumKernel, typename Q = std::decay_t<QuantumKernel>>
std::enable_if_t<std::is_pointer_v<Q> &&
                     std::is_function_v<std::remove_pointer_t<Q>>,
                 QuantumKernel>
createQKernel(QuantumKernel &&kernel) {
  return kernel;
}

// For other non-class, non-function types (if needed)
template <typename QuantumKernel, typename Q = std::decay_t<QuantumKernel>>
std::enable_if_t<!std::is_class_v<Q> &&
                     !(std::is_pointer_v<Q> &&
                       std::is_function_v<std::remove_pointer_t<Q>>)&&!std::
                         is_function_v<std::remove_reference_t<QuantumKernel>>,
                 cudaq::qkernel<Q>>
createQKernel(QuantumKernel &&kernel) {
  return {kernel};
}
#endif

template <typename QuantumKernel>
std::string getKernelName(QuantumKernel &&kernel) {
  if constexpr (has_name<QuantumKernel>::value) {
    // For kernel_builder or objects with .name()
    if constexpr (std::is_lvalue_reference_v<QuantumKernel>) {
      return kernel.name();
    } else {
      static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
      return kernel.name();
    }
  } else {
    // Check if it's a function reference or pointer
    if constexpr (std::is_function_v<std::remove_reference_t<QuantumKernel>> ||
                  (std::is_pointer_v<std::decay_t<QuantumKernel>> &&
                   std::is_function_v<
                       std::remove_pointer_t<std::decay_t<QuantumKernel>>>)) {
      // For function types, use demangling directly
      return __internal__::demangle_kernel(typeid(kernel).name());
    } else {
      // Try registry-based lookup for other types
      auto qKernel =
          cudaq::details::createQKernel(std::forward<QuantumKernel>(kernel));
      auto key = cudaq::registry::__cudaq_getLinkableKernelKey(&qKernel);
      auto name = cudaq::registry::getLinkableKernelNameOrNull(key);
      if (!name) {
        // Fallback
        return __internal__::demangle_kernel(typeid(kernel).name());
      }
      return name;
    }
  }
}
} // namespace details
} // namespace cudaq
