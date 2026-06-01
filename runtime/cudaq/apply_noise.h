/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform.h"
#include "cudaq/qis/qubit_qis.h"

namespace cudaq::details {

template <typename T, typename... RotationT, typename... QuantumT,
          std::size_t NumPProvided = sizeof...(RotationT),
          std::enable_if_t<T::num_parameters == NumPProvided, std::size_t> = 0>
void applyNoiseImpl(const std::tuple<RotationT...> &paramTuple,
                    const std::tuple<QuantumT...> &quantumTuple) {
  auto &platform = get_platform();
  const auto *noiseModel = platform.get_noise();

  // per-spec, no noise model provided, emit warning, no application
  if (!noiseModel)
    return details::warn("apply_noise called but no noise model provided.");

  std::vector<double> parameters;
  cudaq::tuple_for_each(paramTuple,
                        [&](auto &&element) { parameters.push_back(element); });
  std::vector<QuditInfo> qubits;
  cudaq::tuple_for_each(quantumTuple, [&qubits](auto &&element) {
    if constexpr (details::IsQubitType<decltype(element)>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else {
      for (auto &qq : element) {
        qubits.push_back(qubitToQuditInfo(qq));
      }
    }
  });

  if (qubits.size() != T::num_targets) {
    throw std::invalid_argument("Incorrect number of target qubits. Expected " +
                                std::to_string(T::num_targets) + ", got " +
                                std::to_string(qubits.size()));
  }

  auto channel = noiseModel->template get_channel<T>(parameters);
  // per spec - caller provides noise model, but channel not registered,
  // warning generated, no channel application.
  if (channel.empty())
    return;

  getExecutionManager()->applyNoise(channel, qubits);
}
} // namespace cudaq::details

namespace cudaq {

template <typename... Args>
constexpr bool any_float = std::disjunction_v<
    std::is_floating_point<std::remove_cv_t<std::remove_reference_t<Args>>>...>;

#ifdef CUDAQ_REMOTE_SIM
#define TARGET_OK_FOR_APPLY_NOISE false
#else
#define TARGET_OK_FOR_APPLY_NOISE true
#endif

template <typename T, typename... Q>
  requires(std::derived_from<T, cudaq::kraus_channel> && !any_float<Q...> &&
           TARGET_OK_FOR_APPLY_NOISE)
void apply_noise(const std::vector<double> &params, Q &&...args) {
  auto &platform = get_platform();
  const auto *noiseModel = platform.get_noise();

  // per-spec, no noise model provided, emit warning, no application
  if (!noiseModel)
    return details::warn("apply_noise called but no noise model provided. "
                         "skipping kraus channel application.");

  std::vector<QuditInfo> qubits;
  auto argTuple = std::forward_as_tuple(args...);
  cudaq::tuple_for_each(argTuple, [&qubits](auto &&element) {
    if constexpr (details::IsQubitType<decltype(element)>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else {
      for (auto &qq : element) {
        qubits.push_back(qubitToQuditInfo(qq));
      }
    }
  });

  auto channel = noiseModel->template get_channel<T>(params);
  // per spec - caller provides noise model, but channel not registered,
  // warning generated, no channel application.
  if (channel.empty())
    return;
  getExecutionManager()->applyNoise(channel, qubits);
}

class kraus_channel;

template <unsigned len, typename A, typename... As>
constexpr unsigned count_leading_floats() {
  if constexpr (std::is_floating_point_v<std::remove_cvref_t<A>>) {
    return count_leading_floats<len + 1, As...>();
  } else {
    return len;
  }
}
template <unsigned len>
constexpr unsigned count_leading_floats() {
  return len;
}

template <typename... Args>
constexpr bool any_vector_of_float = std::disjunction_v<std::is_same<
    std::vector<double>, std::remove_cv_t<std::remove_reference_t<Args>>>...>;

template <typename T, typename... Args>
  requires(std::derived_from<T, cudaq::kraus_channel> &&
           !any_vector_of_float<Args...> && TARGET_OK_FOR_APPLY_NOISE)
void apply_noise(Args &&...args) {
  constexpr auto ctor_arity = count_leading_floats<0, Args...>();
  constexpr auto qubit_arity = sizeof...(args) - ctor_arity;

  details::applyNoiseImpl<T>(
      details::tuple_slice<ctor_arity>(std::forward_as_tuple(args...)),
      details::tuple_slice_last<qubit_arity>(std::forward_as_tuple(args...)));
}

} // namespace cudaq
