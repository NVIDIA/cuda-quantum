/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <concepts>

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"

namespace cudaq {

namespace __internal__ {

std::string draw(const Trace &trace);

std::string getLaTeXString(const Trace &trace);

} // namespace __internal__

namespace details {

/// @brief execute the kernel functor (with optional arguments) and return the
/// trace of the execution path.
template <typename KernelFunctor, typename... Args>
cudaq::Trace traceFromKernel(KernelFunctor &&kernel, Args &&...args) {
  // Get the platform.
  auto &platform = cudaq::get_platform();

  // This can only be done in simulation
  if (!platform.is_simulator())
    throw std::runtime_error("Cannot use draw on a physical QPU.");

  // Create an execution context, indicate this is for tracing the execution
  // path
  ExecutionContext context("tracer");

  // set the context, execute and then reset
  platform.set_exec_ctx(&context);
  kernel(args...);
  platform.reset_exec_ctx();

  return context.kernelTrace;
}

/// @brief Execute the given kernel functor and extract the
/// state representation.
template <typename KernelFunctor>
std::string extractTrace(KernelFunctor &&kernel) {
  return __internal__::draw(traceFromKernel(kernel));
}

/// @brief Execute the given kernel functor and extract the
/// state representation as LaTeX.
template <typename KernelFunctor>
std::string extractTraceLatex(KernelFunctor &&kernel) {
  return __internal__::getLaTeXString(traceFromKernel(kernel));
}

} // namespace details

// clang-format off
///
/// @brief Returns a drawing of the execution path, i.e., the trace, of the
/// kernel. The drawing is a UTF-8 encoded string.
///
/// \param kernel The quantum callable with non-trivial function signature.
/// \param args The arguments required for evaluation of the quantum kernel.
/// \returns The UTF-8 encoded string of the circuit, without measurement operations.
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/algorithms/draw.h>
/// #include <iostream>
///
/// auto bell_pair = []() __qpu__ {
///     cudaq::qvector q(2);
///     h(q[0]);
///     x<cudaq::ctrl>(q[0], q[1]);
///     mz(q);
/// };
/// ...
/// std::cout << cudaq::draw(bell_pair);
/// /* Output:
///      ╭───╮     
/// q0 : ┤ h ├──●──
///      ╰───╯╭─┴─╮
/// q1 : ─────┤ x ├
///           ╰───╯
/// */
///
/// auto kernel = [](float angle) __qpu__ {
///   cudaq::qvector q(1);
///   h(q[0]);
///   ry(angle, q[0]);
/// };
/// ...
/// std::cout << cudaq::draw(kernel, 0.59);
/// /* Output:
///      ╭───╮╭──────────╮
/// q0 : ┤ h ├┤ ry(0.59) ├
///      ╰───╯╰──────────╯
/// */      
/// \endcode
///
// clang-format on

#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
#else
template <
    typename QuantumKernel, typename... Args,
    typename = std::enable_if_t<std::is_invocable_v<QuantumKernel, Args...>>>
#endif
std::string draw(QuantumKernel &&kernel, Args &&...args) {
  return __internal__::draw(
      details::traceFromKernel(kernel, std::forward<Args>(args)...));
}

#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
#else
template <
    typename QuantumKernel, typename... Args,
    typename = std::enable_if_t<std::is_invocable_v<QuantumKernel, Args...>>>
#endif
std::string draw(std::string format, QuantumKernel &&kernel, Args &&...args) {
  if (format == "ascii") {
    return draw(kernel, std::forward<Args>(args)...);
  } else if (format == "latex") {
    return __internal__::getLaTeXString(
        details::traceFromKernel(kernel, std::forward<Args>(args)...));
  } else {
    throw std::runtime_error(
        "Invalid format. Supported formats are 'ascii' and 'latex'.");
  }
}

/// @brief Outputs the drawing of a circuit to an output stream.
template <typename QuantumKernel, typename... Args>
void draw(std::ostream &os, QuantumKernel &&kernel, Args &&...args) {
  auto drawing = draw(kernel, std::forward<Args>(args)...);
  os << drawing;
}

} // namespace cudaq
