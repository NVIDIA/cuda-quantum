/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/platform.h"
#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qkernel.h"
#include "cudaq/qis/qreg.h"
#include "cudaq/qis/qvector.h"
#include <algorithm>
#include <cstring>
#include <functional>

#define __qpu__ __attribute__((annotate("quantum")))

// This file describes the API for a default qubit logical instruction
// set for CUDA-Q kernels.

namespace cudaq {

namespace details {
void warn(const std::string_view msg);
}

// Define the common single qubit operations.
namespace qubit_op {
#define ConcreteQubitOp(NAME)                                                  \
  struct NAME##Op {                                                            \
    static const std::string name() { return #NAME; }                          \
  };

ConcreteQubitOp(h) ConcreteQubitOp(x) ConcreteQubitOp(y) ConcreteQubitOp(z)
    ConcreteQubitOp(s) ConcreteQubitOp(t) ConcreteQubitOp(rx)
        ConcreteQubitOp(ry) ConcreteQubitOp(rz) ConcreteQubitOp(r1)
            ConcreteQubitOp(u3)

} // namespace qubit_op

// Convert a qubit to its unique id representation
inline QuditInfo qubitToQuditInfo(qubit &q) { return {q.n_levels(), q.id()}; }
inline bool qubitIsNegative(qubit &q) { return q.is_negative(); }

/// C++20 variant using templates.

/// This function will apply the specified `QuantumOp`. It will check the
/// modifier template type and if it is `base`, it will apply the operation to
/// any qubits provided as input. If `ctrl`, it will take the first `N-1` qubits
/// as the controls and the last qubit as the target.
template <typename QuantumOp, typename mod = base, typename... QubitArgs>
void oneQubitApply(QubitArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  // Get the number of input qubits
  constexpr std::size_t nArgs = sizeof...(QubitArgs);

  // Map the qubits to their unique ids and pack them.
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(args)...};
  std::vector<bool> qubitIsNegated{qubitIsNegative(args)...};

  // If mod == base, then we just want to apply the gate to all qubits provided.
  // This is a broadcast application.
  if constexpr (std::is_same_v<mod, base>) {
    for (auto &qubit : quditInfos)
      getExecutionManager()->apply(gateName, {}, {}, {qubit});

    // Nothing left to do, return
    return;
  }

  // If we are here, then `mod` must be control or adjoint. Extract the controls
  // and the target
  std::vector<QuditInfo> controls(quditInfos.begin(),
                                  quditInfos.begin() + nArgs - 1);

  // If we have controls, check if any of them are negative controls, and if so
  // apply an x.
  if (!controls.empty())
    for (std::size_t i = 0; i < controls.size(); i++)
      if (qubitIsNegated[i])
        getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Apply the gate
  getExecutionManager()->apply(gateName, {}, controls, {quditInfos.back()},
                               std::is_same_v<mod, adj>);

  // If we did apply any X ops for a negative control, we need to reverse it.
  if (!controls.empty()) {
    for (std::size_t i = 0; i < controls.size(); i++) {
      if (qubitIsNegated[i]) {
        getExecutionManager()->apply("x", {}, {}, {controls[i]});
        // fold expression which will reverse the negation
        (
            [&] {
              if (args.is_negative())
                args.negate();
            }(),
            ...);
      }
    }
  }
}

/// This function will apply a multi-controlled operation with the given control
/// register on the single qubit target.
template <typename QuantumOp, typename mod = ctrl, typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void oneQubitApplyControlledRange(QubitRange &ctrls, qubit &target) {
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the input control register to a vector of QuditInfo
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });

  // Apply the gate
  getExecutionManager()->apply(gateName, {}, controls,
                               {cudaq::qubitToQuditInfo(target)});
}

#define CUDAQ_QIS_ONE_TARGET_QUBIT_(NAME)                                      \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename mod = base, typename... QubitArgs>                        \
  void NAME(QubitArgs &...args) {                                              \
    oneQubitApply<qubit_op::NAME##Op, mod>(args...);                           \
  }                                                                            \
  template <typename mod = ctrl, typename QubitRange>                          \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(QubitRange &ctrls, qubit &target) {                                \
    oneQubitApplyControlledRange<qubit_op::NAME##Op, mod>(ctrls, target);      \
  }                                                                            \
  template <typename mod = base, typename QubitRange>                          \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(QubitRange &qr) {                                                  \
    for (auto &q : qr) {                                                       \
      NAME<mod>(q);                                                            \
    }                                                                          \
  }                                                                            \
  template <typename mod = base, typename QubitRange>                          \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(QubitRange &&qr) {                                                 \
    for (auto &q : qr) {                                                       \
      NAME<mod>(q);                                                            \
    }                                                                          \
  }

// Instantiate the above 3 functions for the default logical gate set
CUDAQ_QIS_ONE_TARGET_QUBIT_(h)
CUDAQ_QIS_ONE_TARGET_QUBIT_(x)
CUDAQ_QIS_ONE_TARGET_QUBIT_(y)
CUDAQ_QIS_ONE_TARGET_QUBIT_(z)
CUDAQ_QIS_ONE_TARGET_QUBIT_(t)
CUDAQ_QIS_ONE_TARGET_QUBIT_(s)

template <typename QuantumOp, typename mod = base, typename ScalarAngle,
          typename... QubitArgs>
void oneQubitSingleParameterApply(ScalarAngle angle, QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};
  std::vector<double> parameters{static_cast<double>(angle)};

  // If there are more than one qubits and mod == base, then
  // we just want to apply the same gate to all qubits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &targetId : targets)
      getExecutionManager()->apply(gateName, parameters, {}, {targetId});

    // Nothing left to do, return
    return;
  }

  // If we are here, then mod must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply(gateName, parameters, controls, {targets.back()},
                               std::is_same_v<mod, adj>);
}

template <typename QuantumOp, typename mod = ctrl, typename ScalarAngle,
          typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void oneQubitSingleParameterControlledRange(ScalarAngle angle,
                                            QubitRange &ctrls, qubit &target) {
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the input control register to a vector of QuditInfo
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });

  // Apply the gate
  getExecutionManager()->apply(gateName, {angle}, controls,
                               {qubitToQuditInfo(target)});
}

#define CUDAQ_QIS_PARAM_ONE_TARGET_(NAME)                                      \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename mod = base, typename ScalarAngle, typename... QubitArgs>  \
  void NAME(ScalarAngle angle, QubitArgs &...args) {                           \
    oneQubitSingleParameterApply<qubit_op::NAME##Op, mod>(angle, args...);     \
  }                                                                            \
  template <typename mod = ctrl, typename ScalarAngle, typename QubitRange>    \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(ScalarAngle angle, QubitRange &ctrls, qubit &target) {             \
    oneQubitSingleParameterControlledRange<qubit_op::NAME##Op, mod>(           \
        angle, ctrls, target);                                                 \
  }

// FIXME add One Qubit Single Parameter Broadcast over register with an angle
// for each

CUDAQ_QIS_PARAM_ONE_TARGET_(rx)
CUDAQ_QIS_PARAM_ONE_TARGET_(ry)
CUDAQ_QIS_PARAM_ONE_TARGET_(rz)
CUDAQ_QIS_PARAM_ONE_TARGET_(r1)

namespace types {
struct u3 {
  inline static const std::string name{"u3"};
};
} // namespace types

template <typename mod = base, typename ScalarAngle, typename... QubitArgs>
void u3(ScalarAngle theta, ScalarAngle phi, ScalarAngle lambda,
        QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  std::vector<ScalarAngle> parameters{theta, phi, lambda};

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};

  // If there are more than one qubits and mod == base, then
  // we just want to apply the same gate to all qubits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &targetId : targets)
      getExecutionManager()->apply("u3", parameters, {}, {targetId});
    return;
  }

  // If we are here, then mod must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply("u3", parameters, controls, {targets.back()},
                               std::is_same_v<mod, adj>);
}
template <typename mod = ctrl, typename ScalarAngle, typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void u3(ScalarAngle theta, ScalarAngle phi, ScalarAngle lambda,
        QubitRange &ctrls, qubit &target) {
  std::vector<ScalarAngle> parameters{theta, phi, lambda};
  // Map the input control register to a vector of QuditInfo
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });

  // Apply the gate
  getExecutionManager()->apply("u3", parameters, controls,
                               {qubitToQuditInfo(target)});
}

// Define the swap gate instruction and control versions of it
namespace types {
struct swap {
  inline static const std::string name{"swap"};
};
} // namespace types

template <typename mod = base, typename... QubitArgs>
void swap(QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> qubitIds{qubitToQuditInfo(args)...};
  if constexpr (nArgs == 2) {
    getExecutionManager()->apply("swap", {}, {}, qubitIds);
    return;
  } else {
    static_assert(std::is_same_v<mod, ctrl>,
                  "More than 2 qubits passed to swap but modifier != ctrl.");
  }

  // Controls are all qubits except the last 2
  std::vector<QuditInfo> controls(qubitIds.begin(),
                                  qubitIds.begin() + qubitIds.size() - 2);
  std::vector<QuditInfo> targets(qubitIds.end() - 2, qubitIds.end());
  getExecutionManager()->apply("swap", {}, controls, targets);
}

template <typename QuantumRegister>
  requires(std::ranges::range<QuantumRegister>)
void swap(QuantumRegister &ctrls, qubit &src, qubit &target) {
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });
  getExecutionManager()->apply(
      "swap", {}, controls, {qubitToQuditInfo(src), qubitToQuditInfo(target)});
}

// Define common 2 qubit operations.
inline void cnot(qubit &q, qubit &r) { x<cudaq::ctrl>(q, r); }
inline void cx(qubit &q, qubit &r) { x<cudaq::ctrl>(q, r); }
inline void cy(qubit &q, qubit &r) { y<cudaq::ctrl>(q, r); }
inline void cz(qubit &q, qubit &r) { z<cudaq::ctrl>(q, r); }
inline void ch(qubit &q, qubit &r) { h<cudaq::ctrl>(q, r); }
inline void cs(qubit &q, qubit &r) { s<cudaq::ctrl>(q, r); }
inline void ct(qubit &q, qubit &r) { t<cudaq::ctrl>(q, r); }
inline void ccx(qubit &q, qubit &r, qubit &s) { x<cudaq::ctrl>(q, r, s); }

// Define common 2 qubit operations with angles.
template <typename T>
void crx(T angle, qubit &q, qubit &r) {
  rx<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void cry(T angle, qubit &q, qubit &r) {
  ry<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void crz(T angle, qubit &q, qubit &r) {
  rz<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void cr1(T angle, qubit &q, qubit &r) {
  r1<cudaq::ctrl>(angle, q, r);
}

// Define common single qubit adjoint operations.
inline void sdg(qubit &q) { s<cudaq::adj>(q); }
inline void tdg(qubit &q) { t<cudaq::adj>(q); }

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void exp_pauli(double theta, QubitRange &&qubits, const char *pauliWord) {
  std::vector<QuditInfo> quditInfos;
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(quditInfos),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });
  // FIXME: it would be cleaner if we just kept it as a pauli word here
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void exp_pauli(double theta, QubitRange &&qubits,
               const cudaq::pauli_word &pauliWord) {
  exp_pauli(theta, qubits, pauliWord.str().c_str());
}

/// @brief Apply a general Pauli rotation, takes a variadic set of
/// qubits, and the number of qubits must be equal to the Pauli word length.
template <typename... QubitArgs>
void exp_pauli(double theta, const char *pauliWord, QubitArgs &...qubits) {

  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation with control qubits and a variadic set
/// of qubits. The number of qubits must be equal to the Pauli word length.
template <typename QuantumRegister, typename... QubitArgs>
  requires(std::ranges::range<QuantumRegister>)
void exp_pauli(QuantumRegister &ctrls, double theta, const char *pauliWord,
               QubitArgs &...qubits) {
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });
  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, controls, quditInfos,
                               false, spin_op::from_word(pauliWord));
}

/// @brief Measure an individual qubit, return 0,1 as `bool`
inline measure_result mz(qubit &q) {
  return getExecutionManager()->measure(QuditInfo{q.n_levels(), q.id()});
}

/// @brief Measure an individual qubit in `x` basis, return 0,1 as `bool`
inline measure_result mx(qubit &q) {
  h(q);
  return getExecutionManager()->measure(QuditInfo{q.n_levels(), q.id()});
}

// Measure an individual qubit in `y` basis, return 0,1 as `bool`
inline measure_result my(qubit &q) {
  r1(-M_PI_2, q);
  h(q);
  return getExecutionManager()->measure(QuditInfo{q.n_levels(), q.id()});
}

inline void reset(qubit &q) {
  getExecutionManager()->reset({q.n_levels(), q.id()});
}

// Measure all qubits in the range, return vector of 0,1
template <typename QubitRange>
  requires std::ranges::range<QubitRange>
std::vector<measure_result> mz(QubitRange &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.push_back(mz(qq));
  }
  return b;
}

template <std::size_t Levels>
std::vector<measure_result> mz(const qview<Levels> &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.emplace_back(mz(qq));
  }
  return b;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs);

template <typename QubitRange, typename... Qs>
  requires(std::ranges::range<QubitRange>)
std::vector<measure_result> mz(QubitRange &qr, Qs &&...qs) {
  std::vector<measure_result> result = mz(qr);
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs) {
  std::vector<measure_result> result = {mz(q)};
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

namespace support {
// Helpers to deal with the `vector<bool>` specialized template type.
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &, char *,
                                             std::size_t);
void __nvqpp_vector_bool_to_initializer_list(void *, const std::vector<bool> &,
                                             std::vector<char *> **);
void __nvqpp_vector_bool_free_temporary_initlists(std::vector<char *> *);
}
} // namespace support

// Measure the state in the given spin_op basis.
inline SpinMeasureResult measure(const cudaq::spin_op &term) {
  return getExecutionManager()->measure(term);
}

// Cast a measure register to an int64_t.
// This function is classic control code that may run on a QPU.
inline std::int64_t to_integer(const std::vector<measure_result> &bits) {
  std::int64_t ret = 0;
  for (std::size_t i = 0; i < bits.size(); i++) {
    if (bits[i]) {
      ret |= 1UL << i;
    }
  }
  return ret;
}

inline std::int64_t to_integer(const std::string &arg) {
  std::string bitString{arg};
  std::reverse(bitString.begin(), bitString.end());
  return std::stoull(bitString, nullptr, 2);
}

// This concept tests if `Kernel` is a `Callable` that takes the arguments,
// `Args`, and returns `void`.
template <typename Kernel, typename... Args>
concept isCallableVoidKernel = requires(Kernel &&k, Args &&...args) {
  { k(args...) } -> std::same_as<void>;
};

template <typename T, typename Signature>
concept signature = std::is_convertible_v<T, std::function<Signature>>;

template <typename T>
concept takes_qubit = signature<T, void(qubit &)>;

template <typename T>
concept takes_qvector = signature<T, void(qvector<> &)>;

// Control the given cudaq kernel on the given control qubit
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
void control(QuantumKernel &&kernel, qubit &control, Args &&...args) {
  std::vector<std::size_t> ctrls{control.id()};
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given register of control qubits
template <typename QuantumKernel, typename QuantumRegister, typename... Args>
  requires std::ranges::range<QuantumRegister> &&
           isCallableVoidKernel<QuantumKernel, Args...>
void control(QuantumKernel &&kernel, QuantumRegister &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (std::size_t i = 0; i < ctrl_qubits.size(); i++) {
    ctrls.push_back(ctrl_qubits[i].id());
  }
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given list of references to control
// qubits.
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
void control(QuantumKernel &&kernel,
             std::vector<std::reference_wrapper<qubit>> &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (auto &cq : ctrl_qubits) {
    ctrls.push_back(cq.get().id());
  }
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Apply the adjoint of the given cudaq kernel
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
void adjoint(QuantumKernel &&kernel, Args &&...args) {
  // static_assert(true, "adj not implemented yet.");
  getExecutionManager()->startAdjointRegion();
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endAdjointRegion();
}

/// Instantiate this type to affect C A C^dag, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_action {
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
void compute_action(ComputeFunction &&c, ActionFunction &&a) {
  c();
  a();
  adjoint(c);
}

/// Instantiate this type to affect C^dag A C, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_dag_action {
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
void compute_dag_action(ComputeFunction &&c, ActionFunction &&a) {
  adjoint(c);
  a();
  c();
}

/// Helper function to extract a slice of a `std::vector<T>` to be used within
/// CUDA-Q kernels.
template <typename T>
  requires(std::is_arithmetic_v<T>)
std::vector<T> slice_vector(std::vector<T> &original, std::size_t start,
                            std::size_t count) {
  std::vector<double> ret(original.begin() + start,
                          original.begin() + start + count);
  return ret;
}

} // namespace cudaq

#define CUDAQ_MOD_TEMPLATE template <typename mod = base, typename... Args>

namespace cudaq::details {

template <typename T>
using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using IsQubitType = std::is_same<remove_cvref<T>, cudaq::qubit>;

template <typename T>
using IsQvectorType = std::is_same<remove_cvref<T>, cudaq::qvector<>>;

template <typename T>
using IsQviewType = std::is_same<remove_cvref<T>, cudaq::qview<>>;

template <typename T>
using IsQarrayType = std::is_base_of<cudaq::qarray_base, remove_cvref<T>>;
// --------------------------

template <size_t N, typename Tuple, size_t... Indices>
auto tuple_slice_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  return std::make_tuple(std::get<Indices>(std::forward<Tuple>(tuple))...);
}

template <size_t N, typename... Args>
auto tuple_slice(std::tuple<Args...> &&tuple) {
  return tuple_slice_impl<N>(std::forward<std::tuple<Args...>>(tuple),
                             std::make_index_sequence<N>{});
}

template <size_t N, typename Tuple, size_t... Indices>
auto tuple_slice_last_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  constexpr size_t M = std::tuple_size_v<std::remove_reference_t<Tuple>> - N;
  return std::forward_as_tuple(
      std::get<M + Indices>(std::forward<Tuple>(tuple))...);
}

template <size_t N, typename... Args>
auto tuple_slice_last(std::tuple<Args...> &&tuple) {
  return tuple_slice_last_impl<N>(std::forward<std::tuple<Args...>>(tuple),
                                  std::make_index_sequence<N>{});
}

/// @brief Map provided qubit arguments to a vector of QuditInfo.
template <typename... QuantumT>
void qubitsToQuditInfos(const std::tuple<QuantumT...> &quantumTuple,
                        std::vector<QuditInfo> &qubits) {
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        qubits.push_back(qubitToQuditInfo(q));
    }
  });
}

/// @brief Search through the qubit arguments and see which ones are negated.
template <typename... QuantumT>
void findQubitNegations(const std::tuple<QuantumT...> &quantumTuple,
                        std::vector<bool> &qubitIsNegated) {
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      qubitIsNegated.push_back(element.is_negative());
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        qubitIsNegated.push_back(q.is_negative());
    }
    return;
  });
}

// 1. Type traits for container detection
template <typename T>
struct is_fixed_size_container : std::false_type {};

template <>
struct is_fixed_size_container<cudaq::qarray_base> : std::true_type {};

// 2. Compile-time qubit counting logic
template <typename T>
constexpr std::size_t count_qubits_compile_time() {
  if constexpr (details::IsQubitType<T>::value) {
    return 1;
  } else if constexpr (details::IsQarrayType<T>::value) {
    return std::tuple_size<std::decay_t<T>>::value;
  } else {
    return 0; // Dynamic containers handled at runtime
  }
}

template <typename Tuple, std::size_t... Is>
constexpr std::size_t sum_targets_impl(std::index_sequence<Is...>) {
  return (count_qubits_compile_time<std::tuple_element_t<Is, Tuple>>() + ...);
}

// Type trait to check if T has a static constexpr integer 'num_parameters'
template <typename T, typename = void>
struct has_num_parameters : std::false_type {};

template <typename T>
struct has_num_parameters<T, std::void_t<decltype(T::num_parameters)>>
    : std::bool_constant<std::is_integral_v<decltype(T::num_parameters)>> {};

template <typename T>
inline constexpr bool has_num_parameters_v = has_num_parameters<T>::value;

/// @brief Generic quantum operation applicator function. Supports the
/// following signatures for a generic operation name `OP`
/// `OP(qubit(s))`
/// `OP<ctrl>(qubit..., qubit)`
/// `OP<ctrl>(qubits, qubit)`
/// `OP(scalar..., qubit(s))`
/// `OP<ctrl>(scalar..., qubit..., qubit)`
/// `OP<ctrl>(scalar..., qubits, qubit)`
/// `OP<adj>(qubit)`
/// `OP<adj>(scalar..., qubit)`
/// Control qubits can be negated. Compile errors should be thrown
/// for erroneous signatures.
template <typename mod, std::size_t NumT, std::size_t NumP,
          typename... RotationT, typename... QuantumT,
          std::size_t NumPProvided = sizeof...(RotationT),
          std::enable_if_t<NumP == NumPProvided, std::size_t> = 0>
void applyQuantumOperation(const std::string &gateName,
                           const std::tuple<RotationT...> &paramTuple,
                           const std::tuple<QuantumT...> &quantumTuple) {

  std::vector<double> parameters;
  cudaq::tuple_for_each(paramTuple,
                        [&](auto &&element) { parameters.push_back(element); });

  std::vector<QuditInfo> qubits;
  qubitsToQuditInfos(quantumTuple, qubits);

  std::vector<bool> qubitIsNegated;
  findQubitNegations(quantumTuple, qubitIsNegated);

  assert(qubitIsNegated.size() == qubits.size() && "qubit mismatch");

  // Catch the case where we have multi-target broadcast, we don't allow that
  if (std::is_same_v<mod, base> && NumT > 1 && qubits.size() > NumT)
    throw std::runtime_error(
        "cudaq does not support broadcast for multi-qubit operations.");

  // Operation on correct number of targets, no controls, possible broadcast
  if ((std::is_same_v<mod, base> || std::is_same_v<mod, adj>)&&NumT == 1) {
    for (auto &qubit : qubits)
      getExecutionManager()->apply(gateName, parameters, {}, {qubit},
                                   std::is_same_v<mod, adj>);
    return;
  }

  // Partition out the controls and targets
  std::size_t numControls = qubits.size() - NumT;
  std::vector<QuditInfo> targets(qubits.begin() + numControls, qubits.end()),
      controls(qubits.begin(), qubits.begin() + numControls);

  // Apply X for any negations
  for (std::size_t i = 0; i < controls.size(); i++)
    if (qubitIsNegated[i])
      getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Apply the gate
  getExecutionManager()->apply(gateName, parameters, controls, targets,
                               std::is_same_v<mod, adj>);

  // Reverse any negations
  for (std::size_t i = 0; i < controls.size(); i++)
    if (qubitIsNegated[i])
      getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Reset the negations
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      if (element.is_negative())
        element.negate();
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        if (q.is_negative())
          q.negate();
    }
  });
}

template <typename mod, std::size_t NUMT, std::size_t NUMP, typename... Args>
void genericApplicator(const std::string &gateName, Args &&...args) {
  applyQuantumOperation<mod, NUMT, NUMP>(
      gateName, tuple_slice<NUMP>(std::forward_as_tuple(args...)),
      tuple_slice_last<sizeof...(Args) - NUMP>(std::forward_as_tuple(args...)));
}

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
  // auto argTuple = std::forward_as_tuple(args...);
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

// Apply noise with runtime vector of parameters
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

#define __qop__ __attribute__((annotate("user_custom_quantum_operation")))

/// Register a new custom unitary operation providing a unique name,
/// the number of target qubits, the number of rotation parameters (can be 0),
/// and the unitary matrix as a 1D row-major `std::vector<complex>`
/// representation following a MSB qubit ordering.
#define CUDAQ_REGISTER_OPERATION(NAME, NUMT, NUMP, ...)                        \
  namespace cudaq {                                                            \
  struct CONCAT(NAME, _operation) : public ::cudaq::unitary_operation {        \
    std::vector<std::complex<double>>                                          \
    unitary(const std::vector<double> &parameters) const override {            \
      [[maybe_unused]] std::complex<double> i(0, 1.);                          \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    static inline const bool registered_ = []() {                              \
      cudaq::customOpRegistry::getInstance()                                   \
          .registerOperation<CONCAT(NAME, _operation)>(#NAME);                 \
      return true;                                                             \
    }();                                                                       \
  };                                                                           \
  CUDAQ_MOD_TEMPLATE                                                           \
  void NAME(Args &&...args) {                                                  \
    /* Perform registration at call site as well in case the static            \
     * initialization was not executed in the same context, e.g., remote       \
     * execution.*/                                                            \
    cudaq::customOpRegistry::getInstance()                                     \
        .registerOperation<CONCAT(NAME, _operation)>(#NAME);                   \
    details::genericApplicator<mod, NUMT, NUMP>(#NAME,                         \
                                                std::forward<Args>(args)...);  \
  }                                                                            \
  }                                                                            \
  __qop__ std::vector<std::complex<double>> CONCAT(NAME,                       \
                                                   CONCAT(_generator_, NUMT))( \
      const std::vector<double> &parameters = std::vector<double>()) {         \
    return __VA_ARGS__;                                                        \
  }
