/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file policy_cpos.h
/// @brief Customization-point objects (CPOs) for policy-based dispatching.
///
/// Each CPO wraps a named customization point (e.g. @c finalize). To opt in
/// for a policy @c P, declare a hidden-friend overload of the @c finalize
/// function inside the policy struct; the CPO discovers it via ADL.
///
/// When no policy-specific overload is found, the CPO falls back to a
/// default implementation that bypasses the policy entirely.

#pragma once

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq {
class ExecutionManager;
class ExecutionContext;

/// CPO anchor required so that the name exists in cudaq:: before
/// the concept and CPO below are parsed.
void finalize_execution_manager_impl(ExecutionManager &mgr,
                                     ExecutionContext &ctx);

namespace detail {

/// @brief Detects whether a policy-specific @c finalize_execution_manager_impl
///        overload exists for type @p T (found via ADL through hidden friends).
template <class T>
concept has_em_custom_finalize =
    requires(cudaq::ExecutionManager &mgr, const T &policy,
             cudaq::ExecutionContext &ctx) {
      cudaq::finalize_execution_manager_impl(mgr, policy, ctx);
    };

/// @brief CPO function object for ExecutionManager finalization.
///
/// Dispatches to a policy-specific @c finalize_execution_manager_impl if one
/// exists, otherwise falls back to the 2-argument default.
struct finalize_execution_manager_fn {
  template <class Policy>
  decltype(auto) operator()(cudaq::ExecutionManager &mgr, const Policy &policy,
                            cudaq::ExecutionContext &ctx) const {
    if constexpr (has_em_custom_finalize<Policy>) {
      return cudaq::finalize_execution_manager_impl(mgr, policy, ctx);
    } else {
      return cudaq::finalize_execution_manager_impl(mgr, ctx);
    }
  }
};

} // namespace detail

/// @brief CPO: finalize an execution context via the ExecutionManager.
inline constexpr detail::finalize_execution_manager_fn
    finalize_execution_manager{};

} // namespace cudaq

namespace nvqir {

/// CPO anchor required so that the name exists in nvqir:: before
/// the concept and CPO below are parsed.
void finalize_simulation_circuit_impl(CircuitSimulator &sim,
                                      cudaq::ExecutionContext &ctx);

namespace detail {

/// @brief Detects whether a policy-specific
///        @c finalize_simulation_circuit_impl overload exists for type @p T.
template <class T>
concept has_sim_custom_finalize =
    requires(nvqir::CircuitSimulator &sim, const T &policy,
             cudaq::ExecutionContext &ctx) {
      nvqir::finalize_simulation_circuit_impl(sim, policy, ctx);
    };

/// @brief CPO function object for CircuitSimulator finalization.
///
/// Dispatches to a policy-specific @c finalize_simulation_circuit_impl if one
/// exists, otherwise falls back to the 2-argument default.
struct finalize_simulation_circuit_fn {
  template <class Policy>
  decltype(auto) operator()(nvqir::CircuitSimulator &sim, const Policy &policy,
                            cudaq::ExecutionContext &ctx) const {
    if constexpr (has_sim_custom_finalize<Policy>) {
      return nvqir::finalize_simulation_circuit_impl(sim, policy, ctx);
    } else {
      return nvqir::finalize_simulation_circuit_impl(sim, ctx);
    }
  }
};
} // namespace detail

/// @brief CPO: finalize an execution context via the CircuitSimulator.
inline constexpr detail::finalize_simulation_circuit_fn
    finalize_simulation_circuit{};
} // namespace nvqir
