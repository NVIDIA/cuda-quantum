/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/MeasureCounts.h"
#include "cudaq/spin_op.h"
#include "modifiers.h"
#include "qarray.h"
#include "qreg.h"
#include "qvector.h"
#include <cstring>
#include <functional>

#define __qpu__ __attribute__((annotate("quantum")))

// This file describes the API for a default qubit logical instruction
// set for CUDA Quantum kernels.

namespace cudaq {

// Define the common single qubit operations.
namespace qubit_op {
#define ConcreteQubitOp(NAME)                                                  \
  struct NAME##Op {                                                            \
    static const std::string name() { return #NAME; }                          \
  };

ConcreteQubitOp(h) ConcreteQubitOp(x) ConcreteQubitOp(y) ConcreteQubitOp(z)
    ConcreteQubitOp(s) ConcreteQubitOp(t) ConcreteQubitOp(rx)
        ConcreteQubitOp(ry) ConcreteQubitOp(rz) ConcreteQubitOp(r1)

} // namespace qubit_op

// Convert a qubit to its unique id representation
inline QuditInfo qubitToQuditInfo(qubit &q) { return {q.n_levels(), q.id()}; }
inline bool qubitIsNegative(qubit &q) { return q.is_negative(); }

/// @brief This function will apply the specified `QuantumOp`. It will check the
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

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(args)...};
  std::vector<bool> qubitIsNegated{qubitIsNegative(args)...};

  // If there are more than one qubits and mod == base, then
  // we just want to apply the gate to all qubits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &qubit : quditInfos)
      getExecutionManager()->apply(gateName, {}, {}, {qubit});

    // Nothing left to do, return
    return;
  }

  // If we are here, then `mod` must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(quditInfos.begin(),
                                  quditInfos.begin() + nArgs - 1);

  // If we have controls, check if any of them
  // are negative controls, and if so apply an x
  if (!controls.empty())
    for (std::size_t i = 0; i < controls.size(); i++)
      if (qubitIsNegated[i])
        getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Apply the gate
  getExecutionManager()->apply(gateName, {}, controls, {quditInfos.back()},
                               std::is_same_v<mod, adj>);

  // If we did apply any X ops for a negative control,
  // we need to reverse it
  if (!controls.empty())
    for (std::size_t i = 0; i < controls.size(); i++)
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

/// @brief This function will apply a multi-controlled operation with the given
/// control register on the single qubit target.
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

  // If there are more than one qubits and mod == base, then
  // we just want to apply the same gate to all qubits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &targetId : targets)
      getExecutionManager()->apply(gateName, {angle}, {}, {targetId});

    // Nothing left to do, return
    return;
  }

  // If we are here, then mod must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply(gateName, {angle}, controls, {targets.back()},
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

/// @brief Apply a general Pauli rotation, takes a qubit register and the
/// size must be equal to the pauli word length.
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
inline void exp_pauli(double theta, QubitRange &&qubits,
                      const char *pauliWord) {
  std::vector<QuditInfo> quditInfos;
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(quditInfos),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation, takes a variadic set of
/// qubits, and the number of qubits must be equal to the pauli word length.
template <typename... QubitArgs>
inline void exp_pauli(double theta, const char *pauliWord,
                      QubitArgs &...qubits) {

  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Measure an individual qubit, return 0,1 as `bool`
inline measure_result mz(qubit &q) {
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

/// @brief Measure an individual qubit in `x` basis, return 0,1 as `bool`
inline measure_result mx(qubit &q) {
  h(q);
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

// Measure an individual qubit in `y` basis, return 0,1 as `bool`
inline measure_result my(qubit &q) {
  s<adj>(q);
  h(q);
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

inline void reset(qubit &q) {
  getExecutionManager()->reset({q.n_levels(), q.id()});
}

// Measure all qubits in the range, return vector of 0,1
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
std::vector<measure_result> mz(QubitRange &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.push_back(mz(qq));
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
// Helper to initialize a `vector<bool>` data structure.
extern "C" void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &,
                                                        char *, std::size_t);
} // namespace support

// Measure the state in the given spin_op basis.
inline SpinMeasureResult measure(cudaq::spin_op &term) {
  return getExecutionManager()->measure(term);
}

// Cast a measure register to an int64_t.
// This function is classic control code that may run on a QPU.
inline int64_t to_integer(std::vector<measure_result> bits) {
  int64_t ret = 0;
  for (std::size_t i = 0; i < bits.size(); i++) {
    if (bits[i]) {
      ret |= 1UL << i;
    }
  }
  return ret;
}

inline int64_t to_integer(std::string bitString) {
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
concept takes_qreg = signature<T, void(qreg<> &)>;

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
/// CUDA Quantum kernels.
template <typename T>
  requires(std::is_arithmetic_v<T>)
std::vector<T> slice_vector(std::vector<T> &original, std::size_t start,
                            std::size_t count) {
  std::vector<double> ret(original.begin() + start,
                          original.begin() + start + count);
  return ret;
}

} // namespace cudaq
