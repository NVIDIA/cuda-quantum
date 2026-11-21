/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// C++ frequencies prototype: synchronous launches and stateless preparations
// with qarray<N>& parameters. Local simulation supports histogram/int64
// records; the ServerHelper adapter supports a single frequencies -> emit
// program.

#include "common/SampleResult.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/qis/kernel_utils.h"
#include "cudaq/qis/qarray.h"
#include <cstdint>
#include <variant>

namespace cudaq {
/// Opaque kernel-local histogram. Its lifetime is the current launch.
struct histogram;

/// Prototype outcome stream: one record per explicit emit, in execution order.
using outcome = std::variant<sample_result, std::int64_t>;
using outcome_stream = std::vector<outcome>;

namespace detail {
void begin_outcomes();
outcome_stream take_outcomes();
void discard_outcomes() noexcept;
void publish_histogram(sample_result histogram);

template <typename T>
struct preparation_register;
template <typename C, typename Q, typename... A>
struct preparation_register<void (C::*)(Q &, A...)> {
  using type = Q;
};
template <typename C, typename Q, typename... A>
struct preparation_register<void (C::*)(Q &, A...) const> {
  using type = Q;
};
} // namespace detail

/// Compiler intrinsic. The unreachable invocation makes the preparation visible
/// to the C++ call graph; quantum execution is supplied by quake.frequencies.
template <typename Preparation, typename... Args>
histogram *frequencies(Preparation &prepare, std::size_t shots, Args... args) {
  using Register = typename detail::preparation_register<
      decltype(&Preparation::operator())>::type;
  if (false)
    prepare(*static_cast<Register *>(nullptr), args...);
  throw std::logic_error("frequencies requires compiled kernel execution");
}

/// Explicit output intrinsics. Host calls are not supported by this prototype.
inline void emit(histogram *) {
  throw std::logic_error("emit requires compiled kernel execution");
}
inline void emit(std::int64_t) {
  throw std::logic_error("emit requires compiled kernel execution");
}

/// Execute one program and read its explicit outcomes. No shot policy or
/// interpretation of kernel returns participates in this interface.
template <typename Kernel, typename... Args>
outcome_stream launch(Kernel &&kernel, Args &&...args) {
  static_assert(std::is_void_v<std::invoke_result_t<Kernel, Args...>>,
                "launch entry kernels must return void and emit outcomes");
  auto &platform = get_platform();
  run_policy transport;
  transport.kernelName = detail::getKernelName(kernel);
  transport.noiseModel = platform.get_noise();
  ExecutionContext ctx("run", 1);
  ctx.kernelName = transport.kernelName;
  ctx.noiseModel = transport.noiseModel;
  detail::begin_outcomes();
  try {
    detail::launch(transport, 0, ctx, platform, std::forward<Kernel>(kernel),
                   std::forward<Args>(args)...);
    return detail::take_outcomes();
  } catch (...) {
    detail::discard_outcomes();
    throw;
  }
}
} // namespace cudaq

// Outside namespace cudaq so the frontend treats this as an ordinary functor.
namespace cudaq_vnext_detail {
template <typename Preparation, typename... Args>
struct sample_kernel {
#ifdef __clang__
  [[clang::annotate("quantum")]]
#endif
  void operator()(Preparation prepare, std::size_t shots, Args... args) {
    cudaq::emit(cudaq::frequencies(prepare, shots, args...));
  }
};
} // namespace cudaq_vnext_detail

namespace cudaq {
/// Minimal C++ sampling adapter. The complete computation lives in
/// sample_kernel.
template <typename Preparation, typename... Args>
sample_result sample_vnext(std::size_t shots, Preparation prepare,
                           Args... args) {
  auto records =
      launch(cudaq_vnext_detail::sample_kernel<Preparation, Args...>{}, prepare,
             shots, args...);
  if (records.size() != 1 || !std::holds_alternative<sample_result>(records[0]))
    throw std::runtime_error("sample_vnext expected one emitted histogram");
  return std::get<sample_result>(std::move(records[0]));
}
/// Kernel-first convenience with the existing sampling default of 1000 shots.
template <typename Preparation, typename... Args>
  requires requires { &Preparation::operator(); }
sample_result sample_vnext(Preparation prepare, Args... args) {
  return sample_vnext(std::size_t{1000}, prepare, args...);
}
} // namespace cudaq
