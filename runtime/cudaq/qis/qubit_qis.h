/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/MeasureCounts.h"
#include "cudaq/host_config.h"
#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qreg.h"
#include "cudaq/qis/qvector.h"
#include "cudaq/spin_op.h"
#include <cstring>
#include <functional>

#define __qpu__ __attribute__((annotate("quantum")))

// This file describes the API for a default qubit logical instruction
// set for CUDA-Q kernels.

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
            ConcreteQubitOp(u3)

} // namespace qubit_op

// Convert a qubit to its unique id representation
inline QuditInfo qubitToQuditInfo(qubit &q) { return {q.n_levels(), q.id()}; }
inline bool qubitIsNegative(qubit &q) { return q.is_negative(); }

#if CUDAQ_USE_STD20

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

#else // not C++20

/// C++17 variant does NOT use templates. The user must instead use the 'c'
/// prefix form: `cx`, `cy`, `cz`, etc.

template <typename QuantumOp, typename... QubitArgs>
void oneQubitApply(QubitArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(args)...};

  // If there are more than one qubits, then we just want to apply the gate to
  // all qubits provided
  for (auto &qubit : quditInfos)
    getExecutionManager()->apply(gateName, {}, {}, {qubit});
}

template <typename QuantumOp, typename... QubitArgs>
void oneQubitWithControlsApply(QubitArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  // Get the number of input qubits
  constexpr std::size_t nArgs = sizeof...(QubitArgs);

  // Map the qubits to their unique ids and pack them.
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(args)...};
  std::vector<bool> qubitIsNegated{qubitIsNegative(args)...};

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
                               /*adjoint=*/false);

  // If we did apply any X ops for a negative control, we need to reverse it.
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

template <typename QuantumOp, typename... QubitArgs>
void oneQubitWithAdjointControlsApply(QubitArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  // Get the number of input qubits
  constexpr std::size_t nArgs = sizeof...(QubitArgs);

  // Map the qubits to their unique ids and pack them.
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(args)...};
  std::vector<bool> qubitIsNegated{qubitIsNegative(args)...};

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
                               /*adjoint=*/true);

  // If we did apply any X ops for a negative control, we need to reverse it.
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

template <typename QuantumOp, typename QubitRange>
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
  template <typename... QubitArgs>                                             \
  void NAME(QubitArgs &...args) {                                              \
    oneQubitApply<qubit_op::NAME##Op>(args...);                                \
  }                                                                            \
  void c##NAME(qubit &ctrl, qubit &target) {                                   \
    oneQubitWithControlsApply<qubit_op::NAME##Op>(ctrl, target);               \
  }                                                                            \
  template <typename QubitRange,                                               \
            typename = std::enable_if_t<!std::is_same_v<                       \
                std::remove_reference_t<std::remove_cv_t<QubitRange>>,         \
                cudaq::qubit>>>                                                \
  void NAME(QubitRange &qr) {                                                  \
    for (auto &q : qr) {                                                       \
      NAME(q);                                                                 \
    }                                                                          \
  }                                                                            \
  template <typename QubitRange,                                               \
            typename = std::enable_if_t<!std::is_same_v<                       \
                std::remove_reference_t<std::remove_cv_t<QubitRange>>,         \
                cudaq::qubit>>>                                                \
  void NAME(QubitRange &&qr) {                                                 \
    for (auto &q : qr) {                                                       \
      NAME(q);                                                                 \
    }                                                                          \
  }

inline void ccx(qubit &q, qubit &r, qubit &s) {
  oneQubitWithControlsApply<qubit_op::xOp>(q, r, s);
}
inline void sdg(qubit &q) {
  // Note: no controls are present.
  oneQubitWithAdjointControlsApply<qubit_op::sOp>(q);
}
inline void tdg(qubit &q) {
  // Note: no controls are present.
  oneQubitWithAdjointControlsApply<qubit_op::tOp>(q);
}

#endif // not C++20

// Instantiate the above 3 functions for the default logical gate set
CUDAQ_QIS_ONE_TARGET_QUBIT_(h)
CUDAQ_QIS_ONE_TARGET_QUBIT_(x)
CUDAQ_QIS_ONE_TARGET_QUBIT_(y)
CUDAQ_QIS_ONE_TARGET_QUBIT_(z)
CUDAQ_QIS_ONE_TARGET_QUBIT_(t)
CUDAQ_QIS_ONE_TARGET_QUBIT_(s)

#if CUDAQ_USE_STD20

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

#else // not C++20

template <typename QuantumOp, typename ScalarAngle, typename... QubitArgs>
void oneQubitSingleParameterApply(ScalarAngle angle, QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};

  // We just want to apply the same gate to all qubits provided
  for (auto &targetId : targets)
    getExecutionManager()->apply(gateName, std::vector<double>{angle}, {},
                                 {targetId});
}

template <typename QuantumOp, typename ScalarAngle, typename... QubitArgs>
void oneQubitWithControlsSingleParameterApply(ScalarAngle angle,
                                              QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};

  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply(gateName, {angle}, controls, {targets.back()},
                               /*adjoint=*/false);
}

template <
    typename QuantumOp, typename ScalarAngle, typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
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
  template <typename ScalarAngle, typename... QubitArgs>                       \
  void NAME(ScalarAngle angle, QubitArgs &...args) {                           \
    oneQubitSingleParameterApply<qubit_op::NAME##Op>(angle, args...);          \
  }                                                                            \
  template <typename ScalarAngle>                                              \
  void c##NAME(ScalarAngle angle, qubit &ctrl, qubit &target) {                \
    oneQubitWithControlsSingleParameterApply<qubit_op::NAME##Op>(angle, ctrl,  \
                                                                 target);      \
  }

#endif // not C++20

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

#if CUDAQ_USE_STD20
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

#else // not C++20

template <typename ScalarAngle, typename... QubitArgs>
void u3(ScalarAngle theta, ScalarAngle phi, ScalarAngle lambda,
        QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  std::vector<ScalarAngle> parameters{theta, phi, lambda};

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};

  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply("u3", parameters, controls, {targets.back()});
}

template <typename ScalarAngle>
void cu3(ScalarAngle theta, ScalarAngle phi, ScalarAngle lambda, qubit &ctrl,
         qubit &target) {
  std::vector<ScalarAngle> parameters{theta, phi, lambda};
  std::vector<QuditInfo> controls{qubitToQuditInfo(ctrl)};
  std::vector<QuditInfo> targets{qubitToQuditInfo(target)};
  getExecutionManager()->apply("u3", parameters, controls, targets);
}

#endif // not C++20

// Define the swap gate instruction and control versions of it
namespace types {
struct swap {
  inline static const std::string name{"swap"};
};
} // namespace types

#if CUDAQ_USE_STD20
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

#else // not C++20

template <typename Qubit>
void swap(Qubit &src, Qubit &target) {
  static_assert(std::is_same<qubit, Qubit>::value,
                "Cannot operate on a qudit with Levels != 2");
  std::vector<QuditInfo> qubitIds{qubitToQuditInfo(src),
                                  qubitToQuditInfo(target)};
  getExecutionManager()->apply("swap", {}, {}, qubitIds);
}

void cswap(qubit &ctrl, qubit &src, qubit &target) {
  std::vector<QuditInfo> controls{qubitToQuditInfo(ctrl)};
  std::vector<QuditInfo> targets{qubitToQuditInfo(src),
                                 qubitToQuditInfo(target)};
  getExecutionManager()->apply("swap", {}, controls, targets);
}

template <typename QuantumRegister,
          typename = std::enable_if_t<!std::is_same_v<
              std::remove_reference_t<std::remove_cv_t<QuantumRegister>>,
              cudaq::qubit>>>
void swap(QuantumRegister &ctrls, qubit &src, qubit &target) {
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });
  getExecutionManager()->apply(
      "swap", {}, controls, {qubitToQuditInfo(src), qubitToQuditInfo(target)});
}

#endif // not C++20

#if CUDAQ_USE_STD20
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
#endif

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
void exp_pauli(double theta, QubitRange &&qubits, const char *pauliWord) {
  std::vector<QuditInfo> quditInfos;
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(quditInfos),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
void exp_pauli(double theta, QubitRange &&qubits,
               cudaq::pauli_word &pauliWord) {
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
#if CUDAQ_USE_STD20
template <typename QuantumRegister, typename... QubitArgs>
  requires(std::ranges::range<QuantumRegister>)
#else
template <typename QuantumRegister, typename... QubitArgs,
          typename = std::enable_if_t<
              std::is_same_v<std::remove_reference_t<std::remove_cv_t<
                                 decltype(*QuantumRegister().begin())>>,
                             qubit>>>
#endif
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
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

/// @brief Measure an individual qubit in `x` basis, return 0,1 as `bool`
inline measure_result mx(qubit &q) {
  h(q);
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

// Measure an individual qubit in `y` basis, return 0,1 as `bool`
inline measure_result my(qubit &q) {
  r1(-M_PI_2, q);
  h(q);
  return getExecutionManager()->measure({q.n_levels(), q.id()});
}

inline void reset(qubit &q) {
  getExecutionManager()->reset({q.n_levels(), q.id()});
}

// Measure all qubits in the range, return vector of 0,1
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires std::ranges::range<QubitRange>
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
std::vector<measure_result> mz(QubitRange &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.push_back(mz(qq));
  }
  return b;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs);

#if CUDAQ_USE_STD20
template <typename QubitRange, typename... Qs>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange, typename... Qs,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
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
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &, char *,
                                             std::size_t);
void __nvqpp_vector_bool_to_initializer_list(void *, const std::vector<bool> &);
}
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

#if CUDAQ_USE_STD20
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
#endif

// Control the given cudaq kernel on the given control qubit
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel, qubit &control, Args &&...args) {
  std::vector<std::size_t> ctrls{control.id()};
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given register of control qubits
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename QuantumRegister, typename... Args>
  requires std::ranges::range<QuantumRegister> &&
           isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename QuantumRegister, typename... Args,
          typename = std::enable_if_t<
              !std::is_same_v<
                  std::remove_reference_t<std::remove_cv_t<QuantumRegister>>,
                  cudaq::qubit> &&
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
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
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
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
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void adjoint(QuantumKernel &&kernel, Args &&...args) {
  // static_assert(true, "adj not implemented yet.");
  getExecutionManager()->startAdjointRegion();
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endAdjointRegion();
}

/// Instantiate this type to affect C A C^dag, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_action {
#if CUDAQ_USE_STD20
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
#else
template <
    typename ComputeFunction, typename ActionFunction,
    typename = std::enable_if_t<std::is_invocable_r_v<void, ComputeFunction> &&
                                std::is_invocable_r_v<void, ActionFunction>>>
#endif
void compute_action(ComputeFunction &&c, ActionFunction &&a) {
  c();
  a();
  adjoint(c);
}

/// Instantiate this type to affect C^dag A C, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_dag_action {
#if CUDAQ_USE_STD20
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
#else
template <
    typename ComputeFunction, typename ActionFunction,
    typename = std::enable_if_t<std::is_invocable_r_v<void, ComputeFunction> &&
                                std::is_invocable_r_v<void, ActionFunction>>>
#endif
void compute_dag_action(ComputeFunction &&c, ActionFunction &&a) {
  adjoint(c);
  a();
  c();
}

/// Helper function to extract a slice of a `std::vector<T>` to be used within
/// CUDA-Q kernels.
#if CUDAQ_USE_STD20
template <typename T>
  requires(std::is_arithmetic_v<T>)
#else
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
std::vector<T> slice_vector(std::vector<T> &original, std::size_t start,
                            std::size_t count) {
  std::vector<double> ret(original.begin() + start,
                          original.begin() + start + count);
  return ret;
}

} // namespace cudaq

/// For C++17 we can't adhere to the language specification for
/// the operation modifier type. For this case, we drop the modifier
/// template parameter and users have access to a `cNAME` operation for
/// single controlled operations.
#ifdef CUDAQ_USE_STD20
#define CUDAQ_MOD_TEMPLATE template <typename mod = base, typename... Args>
#else
#define CUDAQ_MOD_TEMPLATE template <typename... Args>
#endif

namespace cudaq::details {

// --------------------------
// Useful C++17 compliant concept checks (note we re-implement
// std::remove_cvref since its a C++20 thing)
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

} // namespace cudaq::details

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
  };                                                                           \
  CUDAQ_MOD_TEMPLATE                                                           \
  void NAME(Args &&...args) {                                                  \
    cudaq::getExecutionManager()->registerOperation<CONCAT(NAME, _operation)>( \
        #NAME);                                                                \
    details::genericApplicator<mod, NUMT, NUMP>(#NAME,                         \
                                                std::forward<Args>(args)...);  \
  }                                                                            \
  }                                                                            \
  __qop__ std::vector<std::complex<double>> CONCAT(NAME,                       \
                                                   CONCAT(_generator_, NUMT))( \
      const std::vector<double> &parameters = std::vector<double>()) {         \
    return __VA_ARGS__;                                                        \
  }
