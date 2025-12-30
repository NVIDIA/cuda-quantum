/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include <concepts>
#include <iostream>

namespace cudaq {

namespace __internal__ {

std::string draw(const Trace &trace);

std::string getLaTeXString(const Trace &trace);

} // namespace __internal__

namespace contrib {

/// @brief execute the kernel functor (with optional arguments) and return the
/// trace of the execution path.
template <typename KernelFunctor, typename... Args>
cudaq::Trace traceFromKernel(KernelFunctor &&kernel, Args &&...args) {
  // Get the platform.
  auto &platform = cudaq::get_platform();

  // This is not supported on hardware backends, but we don't want callers to
  // crash on unhandled exceptions.
  if (!platform.is_simulator()) {
    std::cerr << "Warning: `draw` can only be used with a simulator platform. "
              << "Returning an empty trace." << std::endl;
    return Trace();
  }

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
/// std::cout << cudaq::contrib::draw(bell_pair);
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
/// std::cout << cudaq::contrib::draw(kernel, 0.59);
/// /* Output:
///      ╭───╮╭──────────╮
/// q0 : ┤ h ├┤ ry(0.59) ├
///      ╰───╯╰──────────╯
/// */      
/// \endcode
///
/// @note This function is only available when using simulator backends.
// clang-format on

template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string draw(QuantumKernel &&kernel, Args &&...args) {
  return __internal__::draw(
      contrib::traceFromKernel(kernel, std::forward<Args>(args)...));
}

template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string draw(std::string format, QuantumKernel &&kernel, Args &&...args) {
  if (format == "ascii") {
    return draw(kernel, std::forward<Args>(args)...);
  } else if (format == "latex") {
    return __internal__::getLaTeXString(
        contrib::traceFromKernel(kernel, std::forward<Args>(args)...));
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

} // namespace contrib

namespace details {
/// @brief execute the kernel functor (with optional arguments) and return the
/// trace of the execution path.
template <typename KernelFunctor, typename... Args>
[[deprecated("cudaq::details::traceFromKernel is deprecated - please use "
             "cudaq::contrib::traceFromKernel instead.")]] cudaq::Trace
traceFromKernel(KernelFunctor &&kernel, Args &&...args) {
  return contrib::traceFromKernel(kernel, std::forward<Args>(args)...);
}

/// @brief Execute the given kernel functor and extract the
/// state representation.
template <typename KernelFunctor>
[[deprecated("cudaq::details::extractTrace is deprecated - please use "
             "cudaq::contrib::extractTrace instead.")]] std::string
extractTrace(KernelFunctor &&kernel) {
  return __internal__::draw(traceFromKernel(kernel));
}

/// @brief Execute the given kernel functor and extract the
/// state representation as LaTeX.
template <typename KernelFunctor>
[[deprecated("cudaq::details::extractTraceLatex is deprecated - please use "
             "cudaq::contrib::extractTraceLatex instead.")]] std::string
extractTraceLatex(KernelFunctor &&kernel) {
  return __internal__::getLaTeXString(traceFromKernel(kernel));
}

} // namespace details

template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
[[deprecated("cudaq::draw is deprecated - please use "
             "cudaq::contrib::draw instead.")]] std::string
draw(QuantumKernel &&kernel, Args &&...args) {
  return __internal__::draw(
      contrib::traceFromKernel(kernel, std::forward<Args>(args)...));
}

template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
[[deprecated("cudaq::draw is deprecated - please use "
             "cudaq::contrib::draw instead.")]] std::string
draw(std::string format, QuantumKernel &&kernel, Args &&...args) {
  return contrib::draw(format, kernel, std::forward<Args>(args)...);
}

/// @brief Outputs the drawing of a circuit to an output stream.
template <typename QuantumKernel, typename... Args>
[[deprecated("cudaq::draw is deprecated - please use "
             "cudaq::contrib::draw instead.")]] void
draw(std::ostream &os, QuantumKernel &&kernel, Args &&...args) {
  contrib::draw(os, kernel, std::forward<Args>(args)...);
}

} // namespace cudaq
