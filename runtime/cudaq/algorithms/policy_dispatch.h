/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/policies.h"
#include <string_view>
#include <type_traits>
#include <utility>

namespace cudaq::policies {

/// @brief Run-time policy dispatch utilities for name-based overload selection.
///
/// Typical usage:
/// @code
/// // Simple case: all overloads return void or the same type
/// withPolicy(context.name, [&](auto policy) {
///     beginExecutionContext(policy, context);
/// });
///
/// // Heterogeneous return types: use withPolicy + visitResult
/// withPolicy(context.name, [&](auto policy) {
///   visitResult(
///     [&]{ return finalizeExecutionContext(policy, context); },
///     [&](sample_result&&  r) { context.sample_data  = std::move(r.data); },
///     [&](run_result&&     r) { context.exit_code    = r.exit_code; },
///     [&](void_result)        { context.result       = {}; }
///   );
/// });
/// @endcode

// =============================================================================
// withPolicy
// =============================================================================

/// @brief Maps a runtime name to a compile-time policy and invokes a generic
/// callable.
///
/// Performs a linear search over a static registry of known policy names.
/// On a match, the callable is invoked with the corresponding concrete policy
/// type, enabling compile-time overload resolution. If no match is found, the
/// callable is invoked with @c other_policies{} as a fallback.
///
/// Overload resolution follows standard C++ rules: the most derived policy type
/// is preferred. If no specific overload exists for a given policy, the
/// compiler uses the @c other_policies fallback.
///
/// @note A linear search over a static array is used rather than a hash map.
///       For ~10 short string keys, all entries fit in a few cache lines,
///       making linear search faster than the hash computation and pointer
///       indirection of @c std::unordered_map. Place the most frequently used
///       policies first in the registry for best average performance.
///
/// @tparam Func  A generic callable type, typically a lambda taking @c auto
/// policy.
/// @param  name  The runtime name used to select a policy (e.g. @c
/// context.name).
/// @param  func  A generic callable invoked with the resolved policy tag.
///
/// @par Example
/// @code
/// withPolicy(context.name, [&](auto policy) {
///     beginExecutionContext(policy, context);
/// });
/// @endcode
template <typename Func>
void withPolicy(std::string_view name, Func &&func) {
  using FuncRef = std::remove_reference_t<Func> &;
  using Entry = std::pair<std::string_view, void (*)(FuncRef)>;

  // One static array per Func instantiation — initialized once, no heap
  // allocation. To add a new policy, append an entry here and define the policy
  // struct above.
  static const Entry registry[] = {
      {"sample", [](FuncRef f) { f(sample_policy{}); }},
      {"observe", [](FuncRef f) { f(observe_policy{}); }},
  };

  for (auto &[key, dispatch] : registry) {
    if (name == key) {
      dispatch(func);
      return;
    }
  }

  func(other_policies{});
}

// =============================================================================
// visitResult
// =============================================================================

/// @brief Tag type passed to a visitor when the invoked function returns @c
/// void.
///
/// Allows clients to handle the void case explicitly alongside non-void cases
/// in a uniform visitor pattern, rather than requiring a separate code path.
///
/// @par Example
/// @code
/// visitResult(
///     [&]{ return doSomething(policy, context); },
///     [&](some_result&& r) { ... },
///     [&](void_result)     { ... }  // called when doSomething returns void
/// );
/// @endcode
struct void_result {};

namespace detail {

/// @brief Merges multiple callables into a single visitor that dispatches on
/// argument type.
///
/// Inherits @c operator() from each provided type, allowing the compiler to
/// select the most specific overload based on the argument type at the call
/// site.
///
/// @tparam Ts  Callable types (typically lambdas) to merge into the visitor.
template <typename... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

/// @brief Invokes @p func and forwards its result to @p visitor.
///
/// If @p func returns @c void, @c void_result{} is forwarded to @p visitor
/// instead, ensuring a uniform visitor interface regardless of the return type.
///
/// @tparam Visitor  A callable accepting either the return type of @p func or
/// @c void_result.
/// @tparam Func     A zero-argument callable whose return type determines the
/// dispatch.
/// @param  visitor  The visitor to invoke with the result.
/// @param  func     The callable to invoke.
template <typename Visitor, typename Func>
void invokeVisitor(Visitor &&visitor, Func &&func) {
  if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
    std::forward<Func>(func)();
    std::forward<Visitor>(visitor)(void_result{});
  } else {
    std::forward<Visitor>(visitor)(std::forward<Func>(func)());
  }
}

} // namespace detail

/// @brief Invokes a callable and dispatches its return value to a matching
/// handler.
///
/// Builds a visitor from @p handlers using the @c overloaded idiom, then
/// invokes
/// @p func and forwards its return value to the matching handler. If @p func
/// returns @c void, a @c void_result tag is forwarded instead, allowing the
/// client to handle that case explicitly.
///
/// Intended to be used inside a @c withPolicy call to handle heterogeneous
/// return types from policy-dispatched functions.
///
/// @tparam Func      A zero-argument callable, typically a lambda capturing
///                   the policy and context (e.g. @c [&]{ return f(policy,
///                   context); }).
/// @tparam Handlers  Callable types handling each possible return type.
///                   One handler should accept @c void_result to cover
///                   void-returning overloads.
/// @param  func      The callable to invoke (called exactly once).
/// @param  handlers  Lambdas handling each concrete return type.
///
/// @par Example
/// @code
/// withPolicy(context.name, [&](auto policy) {
///   visitResult(
///     [&]{ return finalizeExecutionContext(policy, context); },
///     [&](sample_result&&  r) { context.sample_data  = std::move(r.data); },
///     [&](run_result&&     r) { context.exit_code    = r.exit_code; },
///     [&](observe_result&& r) { context.observations =
///     std::move(r.observations); },
///     [&](void_result)        { context.result       = {}; }
///   );
/// });
/// @endcode
template <typename Func, typename... Handlers>
void visitResult(Func &&func, Handlers &&...handlers) {
  detail::invokeVisitor(detail::overloaded{std::forward<Handlers>(handlers)...},
                        std::forward<Func>(func));
}
} // namespace cudaq::policies
