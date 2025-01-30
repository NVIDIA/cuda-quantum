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

namespace cudaq {

namespace details {
// The span-like structure for the results of a `cudaq::run` kernel run. The
// span is a variable number of typed result values. These values will be stored
// in a contiguous buffer, the start of which is `data`. The size of the buffer
// must be exactly `lengthInBytes` bytes. `lengthInBytes` is an integer multiple
// of the size of the result type of the kernel launched.
struct RunResultSpan {
  void *data;
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
        cudaq::invokeKernel(std::forward(kernel), std::forward<ARGS>(args)...);
      },
      platform, kernelName, shots);

  std::uint64_t end_offset = span.lengthInBytes / sizeof(RESULT);
  return {reinterpret_cast<RESULT *>(span.data),
          reinterpret_cast<RESULT *>(span.data) + end_offset};
}

// FIXME: Provide an async variant of run?

} // namespace cudaq
