/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/SampleResult.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"
#include "cudaq/host_config.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/qis/kernel_utils.h"
#include <cstdint>

extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &, char *,
                                             std::size_t);
}

namespace cudaq {

namespace details {
// The span-like structure for the results of a `cudaq::run` kernel run. The
// span is a variable number of typed result values. These values will be stored
// in a contiguous buffer, the start of which is `data`. The size of the buffer
// must be exactly `lengthInBytes` bytes. `lengthInBytes` is an integer multiple
// of the size of the result type of the kernel launched.
// NB: for a vector of bool, each bool value is stored in a byte.
struct RunResultSpan {
  char *data;
  std::uint64_t lengthInBytes;
};

// The main entry point to launching a kernel, \p kernel, in a `cudaq::run`
// context and getting back a span containing the results. (The kernel is
// logically executed \p shots times, which can result in up to \p shots
// distinct result values. The results are returned in a span, which is a
// pointer to a buffer and the size of that buffer in bytes.
RunResultSpan runTheKernel(std::function<void()> &&kernel,
                           quantum_platform &platform,
                           const std::string &kernel_name,
                           const std::string &original_name, std::size_t shots,
                           std::size_t qpu_id = 0,
                           /* mlir::Operation* */ void *opt_module = nullptr);

// Template to transfer the ownership of the buffer in a RunResultSpan to a
// `std::vector<T>` object. This special code is required because a
// `std::vector<T>` will always construct its own data, and own it, using its
// standard constructors. In this case, we are transferring ownership of a
// buffer to the vector, `result`, and do not want to make a copy.
template <typename T>
void resultSpanToVectorViaOwnership(std::vector<T> &result,
                                    RunResultSpan &spanIn) {
  // Swap vec into a local variable. vec's original content, if any will be
  // reclaimed at the end of this function.
  std::vector<T> deadEnder;
  std::swap(deadEnder, result);

  // Initialize the vector `result` in place and without any data copies.
  if constexpr (std::is_same_v<T, bool>) {
    // std::vector<bool> is a specialization, so we have to call the
    // vector<bool> constructor in this case to pack the bools.
    __nvqpp_initializer_list_to_vector_bool(result, spanIn.data,
                                            spanIn.lengthInBytes);
  } else {
    using raw_vector = struct {
      T *start;
      T *end0;
      T *end1;
    };
    static_assert(sizeof(std::vector<T>) == sizeof(raw_vector) &&
                  "std::vector must use the nominal 3 pointer implementation");
    raw_vector *rawVec = reinterpret_cast<raw_vector *>(&result);
    rawVec->start = reinterpret_cast<T *>(spanIn.data);
    rawVec->end0 = rawVec->end1 =
        reinterpret_cast<T *>(spanIn.data + spanIn.lengthInBytes);
  }

  // Destroy the contents of the span. The caller no longer owns the `data`
  // buffer, the vector `result` does.
  spanIn.data = nullptr;
  spanIn.lengthInBytes = 0;
}

/// Get the `runnable` kernel from a name and cast it to the intended type. The
/// intended type is the kernel's invocation signature sans return type.
#ifndef CUDAQ_LIBRARY_MODE
template <typename R, typename... As>
void (*get_run_entry_point(qkernel<R(As...)>, const std::string &name))(As...) {
  void *runEntry = cudaq::registry::__cudaq_getRunnableKernel(name.c_str());
  return reinterpret_cast<void (*)(As...)>(runEntry);
}
#endif

} // namespace details

/// @brief Run a kernel \p shots number of times and return a `std::vector` of
/// results.
/// @tparam QuantumKernel Quantum kernel type (must return a non-void result)
/// @tparam ...ARGS Quantum kernel argument types
/// @param shots Number of shots to run
/// @param kernel Quantum kernel
/// @param ...args Kernel arguments
/// @return A vector of results
template <typename QuantumKernel, typename... ARGS>
  requires(!std::is_void_v<std::invoke_result_t<std::decay_t<QuantumKernel>,
                                                std::decay_t<ARGS>...>>)
std::vector<
    std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>>
run(std::size_t shots, QuantumKernel &&kernel, ARGS &&...args) {
  if (shots == 0)
    return {};
  using ResultTy =
      std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>;
  std::vector<ResultTy> results;
  auto &platform = get_platform();
#ifdef CUDAQ_LIBRARY_MODE
  auto ctx = std::make_unique<cudaq::ExecutionContext>("run", 1);
  // Direct kernel invocation loop for library mode
  results.reserve(shots);
  for (std::size_t i = 0; i < shots; ++i) {
    platform.set_exec_ctx(ctx.get());
    results.emplace_back(kernel(std::forward<ARGS>(args)...));
    platform.reset_exec_ctx();
  }
#else
  // Launch the kernel in the appropriate context.
  std::string kernelName{details::getKernelName(kernel)};
  details::RunResultSpan span = details::runTheKernel(
      [&]() mutable {
        auto *runKernel =
            details::get_run_entry_point(qkernel{kernel}, kernelName);
        (*runKernel)(std::forward<ARGS>(args)...);
      },
      platform, kernelName, kernelName, shots);
  details::resultSpanToVectorViaOwnership<ResultTy>(results, span);
#endif
  return results;
}

/// @brief Run a kernel \p shots number of times with noise and return a
/// `std::vector` of results.
/// @tparam QuantumKernel Quantum kernel type (must return a non-void result)
/// @tparam ...ARGS Quantum kernel argument types
/// @param shots Number of shots to run
/// @param noise_model Noise model to use for noisy simulation
/// @param kernel Quantum kernel
/// @param ...args Kernel arguments
/// @return A vector of results
template <typename QuantumKernel, typename... ARGS>
  requires(!std::is_void_v<std::invoke_result_t<std::decay_t<QuantumKernel>,
                                                std::decay_t<ARGS>...>>)
std::vector<
    std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>>
run(std::size_t shots, cudaq::noise_model &noise_model, QuantumKernel &&kernel,
    ARGS &&...args) {
  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().isRemoteSimulator ||
      platform.is_remote())
    throw std::runtime_error(
        "Noise model is not supported on remote platforms.");
  if (shots == 0)
    return {};
  using ResultTy =
      std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>;
  std::vector<ResultTy> results;
#ifdef CUDAQ_LIBRARY_MODE
  // Direct kernel invocation loop for library mode
  platform.set_noise(&noise_model);
  auto ctx = std::make_unique<cudaq::ExecutionContext>("run", 1);
  results.reserve(shots);
  for (std::size_t i = 0; i < shots; ++i) {
    platform.set_exec_ctx(ctx.get());
    results.emplace_back(kernel(std::forward<ARGS>(args)...));
    platform.reset_exec_ctx();
  }
  platform.reset_noise();
#else
  // Launch the kernel in the appropriate context.
  platform.set_noise(&noise_model);
  std::string kernelName{details::getKernelName(kernel)};
  details::RunResultSpan span = details::runTheKernel(
      [&]() mutable {
        auto *runKernel =
            details::get_run_entry_point(qkernel{kernel}, kernelName);
        (*runKernel)(std::forward<ARGS>(args)...);
      },
      platform, kernelName, kernelName, shots);
  platform.reset_noise();
  details::resultSpanToVectorViaOwnership<ResultTy>(results, span);
#endif
  return results;
}

/// @brief Launch a run of a kernel for \p shots number of times on a specific
/// QPU
/// @tparam QuantumKernel Quantum kernel type (must return a non-void result)
/// @tparam ...ARGS Quantum kernel argument types
/// @param qpu_id QPU to launch
/// @param shots Number of shots to run
/// @param kernel Quantum kernel
/// @param ...args Kernel arguments
/// @return A handle (`std::future`) to a vector of results
template <typename QuantumKernel, typename... ARGS>
  requires(!std::is_void_v<std::invoke_result_t<std::decay_t<QuantumKernel>,
                                                std::decay_t<ARGS>...>>)
std::future<std::vector<
    std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>>>
run_async(std::size_t qpu_id, std::size_t shots, QuantumKernel &&kernel,
          ARGS &&...args) {
  auto &platform = cudaq::get_platform();

  if (qpu_id >= platform.num_qpus())
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");

  // Launch the kernel in the appropriate context.
  using ResultTy =
      std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>;
  std::promise<std::vector<ResultTy>> promise;
  auto fut = promise.get_future();
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), qpu_id, shots, &platform, &kernel,
       ... args = std::forward<ARGS>(args)]() mutable {
        if (shots == 0) {
          p.set_value({});
          return;
        }
#ifdef CUDAQ_LIBRARY_MODE
        // Direct kernel invocation loop for library mode
        std::vector<ResultTy> res;
        auto ctx = std::make_unique<cudaq::ExecutionContext>("run", 1, qpu_id);
        res.reserve(shots);
        for (std::size_t i = 0; i < shots; ++i) {
          platform.set_exec_ctx(ctx.get());
          res.emplace_back(kernel(std::forward<ARGS>(args)...));
          platform.reset_exec_ctx();
        }
        p.set_value(std::move(res));
#else
        const std::string kernelName{details::getKernelName(kernel)};
        details::RunResultSpan span = details::runTheKernel(
            [&]() mutable {
              auto *runKernel =
                  details::get_run_entry_point(qkernel{kernel}, kernelName);
              (*runKernel)(std::forward<ARGS>(args)...);
            },
            platform, kernelName, kernelName, shots, qpu_id);
        std::vector<ResultTy> results;
        details::resultSpanToVectorViaOwnership<ResultTy>(results, span);
        p.set_value(std::move(results));
#endif
      });
  platform.enqueueAsyncTask(qpu_id, wrapped);
  return fut;
}

/// @brief Launch a run of a kernel for \p shots number of times with noise on a
/// specific QPU
/// @tparam QuantumKernel Quantum kernel type (must return a non-void result)
/// @tparam ...ARGS Quantum kernel argument types
/// @param qpu_id QPU to launch
/// @param shots Number of shots to run
/// @param noise_model Noise model to use for noisy simulation
/// @param kernel Quantum kernel
/// @param ...args Kernel arguments
/// @return A handle (`std::future`) to a vector of results
template <typename QuantumKernel, typename... ARGS>
  requires(!std::is_void_v<std::invoke_result_t<std::decay_t<QuantumKernel>,
                                                std::decay_t<ARGS>...>>)
std::future<std::vector<
    std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>>>
run_async(std::size_t qpu_id, std::size_t shots,
          cudaq::noise_model &noise_model, QuantumKernel &&kernel,
          ARGS &&...args) {
  auto &platform = cudaq::get_platform();

  if (qpu_id >= platform.num_qpus())
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");
  if (platform.get_remote_capabilities().isRemoteSimulator ||
      platform.is_remote())
    throw std::runtime_error(
        "Noise model is not supported on remote platforms.");
  // Launch the kernel in the appropriate context.
  using ResultTy =
      std::invoke_result_t<std::decay_t<QuantumKernel>, std::decay_t<ARGS>...>;
  std::promise<std::vector<ResultTy>> promise;
  auto fut = promise.get_future();
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), qpu_id, shots, &noise_model, &platform, &kernel,
       ... args = std::forward<ARGS>(args)]() mutable {
        if (shots == 0) {
          p.set_value({});
          return;
        }
#ifdef CUDAQ_LIBRARY_MODE
        // Direct kernel invocation loop for library mode
        platform.set_noise(&noise_model);
        auto ctx = std::make_unique<cudaq::ExecutionContext>("run", 1, qpu_id);
        std::vector<ResultTy> res;
        res.reserve(shots);
        for (std::size_t i = 0; i < shots; ++i) {
          platform.set_exec_ctx(ctx.get());
          res.emplace_back(kernel(std::forward<ARGS>(args)...));
          platform.reset_exec_ctx();
        }
        platform.reset_noise();
        p.set_value(std::move(res));
#else
        platform.set_noise(&noise_model);
        const std::string kernelName{details::getKernelName(kernel)};
        details::RunResultSpan span = details::runTheKernel(
            [&]() mutable {
              auto *runKernel =
                  details::get_run_entry_point(qkernel{kernel}, kernelName);
              (*runKernel)(std::forward<ARGS>(args)...);
            },
            platform, kernelName, kernelName, shots, qpu_id);
        platform.reset_noise();
        std::vector<ResultTy> results;
        details::resultSpanToVectorViaOwnership<ResultTy>(results, span);
        p.set_value(std::move(results));
#endif
      });
  platform.enqueueAsyncTask(qpu_id, wrapped);
  return fut;
}
} // namespace cudaq
