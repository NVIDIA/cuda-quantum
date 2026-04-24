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

/// @brief Default finalization — called when no policy-specific overload
///        exists. Defined in execution_manager.h.
void finalize_execution_manager_impl(cudaq::ExecutionManager &mgr,
                                     cudaq::ExecutionContext &ctx);

namespace detail {

/// @brief Detects whether a policy-specific @c finalize_execution_manager_impl
///        overload exists for type @p T (found via ADL through hidden friends).
template <class T>
concept has_em_custom_finalize =
    requires(cudaq::ExecutionManager &mgr, const T &policy,
             cudaq::ExecutionContext &ctx) {
      finalize_execution_manager_impl(mgr, policy, ctx);
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
      return finalize_execution_manager_impl(mgr, policy, ctx);
    } else {
      return finalize_execution_manager_impl(mgr, ctx);
    }
  }
};

} // namespace detail

/// @brief CPO: finalize an execution context via the ExecutionManager.
inline constexpr detail::finalize_execution_manager_fn
    finalize_execution_manager{};

} // namespace cudaq

namespace nvqir {

/// @brief Default finalization — called when no policy-specific overload
///        exists. Defined in CircuitSimulator.h.
void finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                      cudaq::ExecutionContext &ctx);

namespace detail {

/// @brief Detects whether a policy-specific
///        @c finalize_simulation_circuit_impl overload exists for type @p T.
template <class T>
concept has_sim_custom_finalize =
    requires(nvqir::CircuitSimulator &sim, const T &policy,
             cudaq::ExecutionContext &ctx) {
      finalize_simulation_circuit_impl(sim, policy, ctx);
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
      return finalize_simulation_circuit_impl(sim, policy, ctx);
    } else {
      return finalize_simulation_circuit_impl(sim, ctx);
    }
  }
};
} // namespace detail

/// @brief CPO: finalize an execution context via the CircuitSimulator.
inline constexpr detail::finalize_simulation_circuit_fn
    finalize_simulation_circuit{};
} // namespace nvqir
