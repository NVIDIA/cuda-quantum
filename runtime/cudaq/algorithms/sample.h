/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/MeasureCounts.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"

namespace cudaq {
bool kernelHasConditionalFeedback(const std::string &);
namespace __internal__ {
bool isKernelGenerated(const std::string &);
}
/// @brief Return type for asynchronous sampling.
using async_sample_result = async_result<sample_result>;

/// @brief Define a combined sample function validation concept.
/// These concepts provide much better error messages than old-school SFINAE
template <typename QuantumKernel, typename... Args>
concept SampleCallValid =
    ValidArgumentsPassed<QuantumKernel, Args...> &&
    HasVoidReturnType<std::invoke_result_t<QuantumKernel, Args...>>;

namespace details {

/// @brief Take the input KernelFunctor (a lambda that captures runtime args and
/// invokes the quantum kernel) and invoke the sampling process.
template <typename KernelFunctor>
std::optional<sample_result>
runSampling(KernelFunctor &&wrappedKernel, quantum_platform &platform,
            const std::string &kernelName, int shots, std::size_t qpu_id = 0,
            details::future *futureResult = nullptr,
            std::size_t batchIteration = 0, std::size_t totalBatchIters = 0) {
  // Create the execution context.
  auto ctx = std::make_unique<ExecutionContext>("sample", shots);
  ctx->kernelName = kernelName;

  ctx->batchIteration = batchIteration;
  ctx->totalIterations = totalBatchIters;

  // Tell the context if this quantum kernel has
  // conditionals on measure results
  ctx->hasConditionalsOnMeasureResults =
      cudaq::kernelHasConditionalFeedback(kernelName);

#ifdef CUDAQ_LIBRARY_MODE
  // If we have a kernel that has its quake code registered, we
  // won't check for if statements with the tracer.
  auto isRegistered = cudaq::__internal__::isKernelGenerated(kernelName);

  // One extra check to see if we have mid-circuit
  // measures in library mode
  if (!isRegistered && !ctx->hasConditionalsOnMeasureResults) {
    // Trace the kernel function
    ExecutionContext context("tracer");
    auto &platform = get_platform();
    platform.set_exec_ctx(&context);
    wrappedKernel();
    platform.reset_exec_ctx();
    // In trace mode, if we have a measure result
    // that is passed to an if statement, then
    // we'll have collected registernames
    if (!context.registerNames.empty()) {
      // append new register names to the main sample context
      for (std::size_t i = 0; i < context.registerNames.size(); ++i)
        ctx->registerNames.emplace_back("auto_register_" + std::to_string(i));

      ctx->hasConditionalsOnMeasureResults = true;
    }
  }
#endif

  // Indicate that this is an async exec
  ctx->asyncExec = futureResult != nullptr;

  // Set the platform and the qpu id.
  platform.set_exec_ctx(ctx.get(), qpu_id);
  platform.set_current_qpu(qpu_id);
  auto hasCondFeedback = platform.supports_conditional_feedback();

  // If no conditionals, nothing special to do for library mode
  if (!ctx->hasConditionalsOnMeasureResults) {
    // Execute
    wrappedKernel();

    // If we have a non-null future, set it and return
    if (futureResult) {
      *futureResult = ctx->futureResult;
      return std::nullopt;
    }

    // otherwise lets reset the context and set the data
    platform.reset_exec_ctx(qpu_id);
    return ctx->result;
  }

  // If the execution backend does not support
  // sampling with cond feedback, we'll emulate it here
  if (!hasCondFeedback) {
    sample_result counts;

    // If it has conditionals, loop over individual circuit executions
    for (auto &i : cudaq::range(shots)) {
      // Run the kernel
      wrappedKernel();
      // Reset the context and get the single measure result,
      // add it to the sample_result and clear the context result
      platform.reset_exec_ctx(qpu_id);
      counts += ctx->result;
      ctx->result.clear();
      // Reset the context for the next round,
      // don't need to reset on the last exec
      if (i < static_cast<unsigned>(shots) - 1)
        platform.set_exec_ctx(ctx.get(), qpu_id);
    }

    return counts;
  }

  // At this point, the kernel has conditional
  // feedback, but the backend supports it, so
  // just run the kernel, context will get the sampling results
  wrappedKernel();
  // If we have a non-null future, set it and return
  if (futureResult) {
    *futureResult = ctx->futureResult;
    return std::nullopt;
  }

  platform.reset_exec_ctx(qpu_id);
  return ctx->result;
}

/// @brief Take the input KernelFunctor (a lambda that captures runtime args and
/// invokes the quantum kernel) and invoke the sampling process asynchronously.
/// Return a async_sample_result, clients can retrieve the results at a later
/// time via the `get()` call.
template <typename KernelFunctor>
auto runSamplingAsync(KernelFunctor &&wrappedKernel, quantum_platform &platform,
                      const std::string &kernelName, int shots,
                      std::size_t qpu_id = 0) {
  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");
  }

  // If we are remote, then create the sampling executor with cudaq::future
  // provided
  if (platform.is_remote(qpu_id)) {
    details::future futureResult;
    details::runSampling(std::forward<KernelFunctor>(wrappedKernel), platform,
                         kernelName, shots, qpu_id, &futureResult);
    return async_sample_result(std::move(futureResult));
  }

  // Otherwise we'll create our own future/promise and return it
  KernelExecutionTask task(
      [qpu_id, shots, kernelName, &platform,
       kernel = std::forward<KernelFunctor>(wrappedKernel)]() mutable {
        return details::runSampling(kernel, platform, kernelName, shots, qpu_id)
            .value();
      });

  return async_sample_result(
      details::future(platform.enqueueAsyncTask(qpu_id, task)));
}
} // namespace details

/// \brief Sample the given quantum kernel expression and return the
/// mapping of observed bit strings to corresponding number of
/// times observed.
///
/// \param kernel the kernel expression, must contain final measurements
/// \param args the variadic concrete arguments for evaluation of the kernel.
/// \returns counts, The counts dictionary.
///
/// \details Given a quantum kernel with void return type, sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
sample_result sample(QuantumKernel &&kernel, Args &&...args) {

  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(1000);
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runSampling(
             [&kernel, ... args = std::forward<Args>(args)]() mutable {
               kernel(std::forward<Args>(args)...);
             },
             platform, kernelName, shots)
      .value();
}

/// \brief Sample the given quantum kernel expression and return the
/// mapping of observed bit strings to corresponding number of
/// times observed. Specify the number of shots.
///
/// \param shots the number of samples to collect.
/// \param kernel the kernel expression, must contain final measurements
/// \param args the variadic concrete arguments for evaluation of the kernel.
/// \returns counts, The counts dictionary.
///
/// \details Given a quantum kernel with void return type, sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
auto sample(std::size_t shots, QuantumKernel &&kernel, Args &&...args) {

  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runSampling(
             [&kernel, ... args = std::forward<Args>(args)]() mutable {
               kernel(std::forward<Args>(args)...);
             },
             platform, kernelName, shots)
      .value();
}

/// \brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed.
///
/// \param qpu_id the id of the QPU to run asynchronously on
/// \param kernel the kernel expression, must contain final measurements
/// \param args the variadic concrete arguments for evaluation of the kernel.
/// \returns counts future, A std::future containing the resultant counts
/// dictionary.
///
/// \details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
async_sample_result sample_async(const std::size_t qpu_id,
                                 QuantumKernel &&kernel, Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(1000);
  auto kernelName = cudaq::getKernelName(kernel);

  return details::runSamplingAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      platform, kernelName, shots, qpu_id);
}

/// \brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed.
///
/// \param shots the number of samples to collect
/// \param qpu_id the id of the QPU to run asynchronously on
/// \param kernel the kernel expression, must contain final measurements
/// \param args the variadic concrete arguments for evaluation of the kernel.
/// \returns counts future, A std::future containing the resultant counts
/// dictionary.
///
/// \details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
async_sample_result sample_async(std::size_t shots, std::size_t qpu_id,
                                 QuantumKernel &&kernel, Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

  return details::runSamplingAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      platform, kernelName, shots, qpu_id);
}

/// \brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed. Defaults to the 0th QPU id.
///
/// \param kernel the kernel expression, must contain final measurements
/// \param args the variadic concrete arguments for evaluation of the kernel.
/// \returns counts future, A std::future containing the resultant counts
/// dictionary.
///
/// \details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
auto sample_async(QuantumKernel &&kernel, Args &&...args) {
  return sample_async(0, std::forward<QuantumKernel>(kernel),
                      std::forward<Args>(args)...);
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature void(Args...), this
/// function takes as input a set of vector<Arg>..., a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the ith element of each vector is used ith
/// execution of the standard sample function. Results are collected
/// from the execution of every argument set and returned.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
std::vector<sample_result> sample_n(QuantumKernel &&kernel,
                                    ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of qpus
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto shots = platform.get_shots().value_or(1000);
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret =
        details::runSampling(
            [&kernel, ... args = std::forward<decltype(singleIterParameters)>(
                          singleIterParameters)]() mutable { kernel(args...); },
            platform, kernelName, shots, qpuId, nullptr, counter, N)
            .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature void(Args...), this
/// function takes as input a set of vector<Arg>..., a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the ith element of each vector is used ith
/// execution of the standard sample function. Results are collected
/// from the execution of every argument set and returned. This overload
/// allows the number of circuit executions (shots) to be specified.
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
std::vector<sample_result> sample_n(std::size_t shots, QuantumKernel &&kernel,
                                    ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of qpus
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret =
        details::runSampling(
            [&kernel, ... args = std::forward<decltype(singleIterParameters)>(
                          singleIterParameters)]() mutable { kernel(args...); },
            platform, kernelName, shots, qpuId, nullptr, counter, N)
            .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);
}
} // namespace cudaq
