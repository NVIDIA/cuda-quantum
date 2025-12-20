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
#include "cudaq/utils/registry.h"
#include "qkernel.h"

namespace cudaq::details {

#ifdef CUDAQ_LIBRARY_MODE
template <typename QuantumKernel>
QuantumKernel createQKernel(QuantumKernel &&kernel) {
  return kernel;
}
#else

template <typename T>
using remove_cvref_t = typename std::remove_cvref_t<T>;

template <typename QuantumKernel, typename Q = remove_cvref_t<QuantumKernel>,
          typename Operator = typename cudaq::qkernel_deduction_guide_helper<
              decltype(&Q::operator())>::type,
          std::enable_if_t<std::is_class_v<Q>, bool> = true>
cudaq::qkernel<Operator> createQKernel(QuantumKernel &&kernel) {
  return {kernel};
}

template <typename QuantumKernel, typename Q = remove_cvref_t<QuantumKernel>,
          std::enable_if_t<!std::is_class_v<Q>, bool> = true>
cudaq::qkernel<Q> createQKernel(QuantumKernel &&kernel) {
  return {kernel};
}
#endif

template <typename QuantumKernel>
std::string getKernelName(QuantumKernel &&kernel) {
  if constexpr (has_name<QuantumKernel>::value) {
    // kernel_builder kernel: need to JIT code to get it registered.
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
    return kernel.name();
  } else {
    // R (S::operator())(Args..) or R(*)(Args...) kernels are registered
    // and made linkable in GenDeviceCodeLoader pass.
    auto qKernel =
        cudaq::details::createQKernel(std::forward<QuantumKernel>(kernel));
    auto key = cudaq::registry::__cudaq_getLinkableKernelKey(&qKernel);
    auto name = cudaq::registry::getLinkableKernelNameOrNull(key);
    if (!name)
      return __internal__::demangle_kernel(typeid(kernel).name());
    return name;
  }
}

} // namespace cudaq::details
