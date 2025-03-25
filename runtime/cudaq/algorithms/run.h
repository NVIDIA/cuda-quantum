/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/KernelWrapper.h"
#include "common/MeasureCounts.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"
#include "cudaq/host_config.h"
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
                           const std::string &kernel_name, std::size_t shots);

// Template to transfer the ownership of the buffer in a RunResultSpan to a
// `std::vector<T>` object. This special code is required because a
// `std::vector<T>` will always construct its own data, and own it, using its
// standard constructors. In this case, we are transferring ownership of a
// buffer to the vector, `result`, and do not want to make a copy.
template <typename T>
void resultSpanToVectorViaOwnership(std::vector<T> &result,
                                    RunResultSpan &spanIn) {
  using raw_vector = struct {
    T *start;
    T *end0;
    T *end1;
  };
  static_assert(sizeof(std::vector<T>) == sizeof(raw_vector) &&
                "std::vector must use the nominal 3 pointer implementation");

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

} // namespace details

/// cudaq::run allows an entry-point kernel to be executed a \p shots number of
/// times and return a `std::vector` of results.
template <typename RESULT, typename... ARGS>
#if CUDAQ_USE_STD20
  requires(!std::is_void_v<RESULT>)
#endif
std::vector<RESULT> run(std::size_t shots,
                        std::function<RESULT(ARGS...)> &&kernel,
                        ARGS &&...args) {
  if (shots == 0)
    return {};

  // Launch the kernel in the appropriate context.
  auto &platform = cudaq::get_platform();
  std::string kernelName{cudaq::getKernelName(kernel)};
  details::RunResultSpan span = details::runTheKernel(
      [&]() mutable {
        cudaq::invokeKernel(std::move(kernel), std::forward<ARGS>(args)...);
      },
      platform, kernelName, shots);

  return {reinterpret_cast<RESULT *>(span.data),
          reinterpret_cast<RESULT *>(span.data + span.lengthInBytes)};
}

// FIXME: Provide an async variant of run?

} // namespace cudaq
