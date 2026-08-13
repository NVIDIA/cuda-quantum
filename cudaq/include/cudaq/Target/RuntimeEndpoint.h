
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/KernelArgs.h"
#include "cudaq/Support/Tuple.h"
#include "cudaq/algorithms/policies.h"
#include <any>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cudaq {

class QPU;
class CompiledModule;

namespace detail {

template <launch_policy Policy>
using launch_fn_type = Policy::result_type (*)(std::any &impl,
                                               const Policy &policy,
                                               const CompiledModule &module,
                                               KernelArgs args);

/// A table of launch function pointers, one per policy in @c Policies.
///
/// Given the canonical policy tuple, this holds a heterogeneous tuple of
/// @c launch_fn_type<Policy> and exposes a type-keyed @c get accessor.
template <typename Policies>
class DispatchTable;

template <launch_policy... Ps>
class DispatchTable<std::tuple<Ps...>> {
public:
  DispatchTable() = default;

  /// Build the table by invoking @p factory once per policy. @p factory must be
  /// a callable with an explicit template parameter:
  /// ```
  /// auto table = DispatchTable::create([]<typename Policy> {
  ///   ... something that returns a launch_fn_type<Policy>
  /// });
  /// ```
  template <typename Factory>
  static DispatchTable create(Factory factory) {
    return DispatchTable{table_type{factory.template operator()<Ps>()...}};
  }

  /// Retrieve the launch function pointer registered for @p Policy.
  template <launch_policy Policy>
  launch_fn_type<Policy> get() const {
    return std::get<find_pos_v<Policy, std::tuple<Ps...>>>(fns);
  }

  /// Register the launch function pointer for @p Policy.
  template <launch_policy Policy>
  void set(launch_fn_type<Policy> fn) {
    std::get<find_pos_v<Policy, std::tuple<Ps...>>>(fns) = fn;
  }

private:
  using table_type = std::tuple<launch_fn_type<Ps>...>;

  explicit DispatchTable(table_type fns) : fns(std::move(fns)) {}

  table_type fns{};
};

} // namespace detail

struct RuntimeEndpoint {
  /// Launch function pointers for all supported policies, keyed by policy type.
  detail::DispatchTable<all_policies> dispatch;

  /// Store any RuntimeEndpoint state here. Passed by mutable reference to each
  /// launch invocation.
  std::any impl;

  template <launch_policy Policy>
  typename Policy::result_type launchKernel(const Policy &policy,
                                            const CompiledModule &module,
                                            KernelArgs args) {
    auto fn = dispatch.get<Policy>();
    if (!fn) {
      throw std::runtime_error(std::string("Unsupported policy: '") +
                               get_policy_name(policy) + "'");
    }
    return fn(impl, policy, module, args);
  }

  /// Create a RuntimeEndpoint from a QPU instance.
  static RuntimeEndpoint wrapQPU(QPU &qpu);
};

} // namespace cudaq
