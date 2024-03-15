/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cudaq.h>
#include <cudaq/utils/cudaq_utils.h>

namespace cudaq {
namespace internal {

/// Applies the doubly controlled–NOT (CCNOT) gate to three qubits via
/// decomposition.
void CCNOT(qubit &a, qubit &b, qubit &c) __qpu__ {
  h(c);
  cx(b, c);
  t<cudaq::adj>(c);
  cx(a, c);
  t(c);
  cx(b, c);
  t<cudaq::adj>(c);
  cx(a, c);
  t(b);
  t(c);
  h(c);
  cx(a, b);
  t(a);
  t<cudaq::adj>(b);
  cx(a, b);
}

/// Collects the given list of control qubits into one or two of the given
/// auxiliary qubits, using all but the last qubits in the auxiliary list as
/// scratch qubits.
/// For example, if the controls list is 6 qubits, the auxiliary list must be 5
/// qubits, and the state from the 6 control qubits will be collected into the
/// last qubit of the auxiliary array.
/// The adjustment is used to allow the caller to reduce or increase
/// the number of times this is run based on the eventual number of control
/// qubits needed.
void CollectControls(
    const std::vector<std::reference_wrapper<cudaq::qubit>> &ctls,
    cudaq::qview<> aux, int adjustment) __qpu__ {
  for (int i = 0; i < ctls.size() - 1; i += 2) {
    CCNOT(ctls[i], ctls[i + 1], aux[i / 2]);
  }
  for (int i = 0; i < ctls.size() / 2 - 1 - adjustment; ++i) {
    CCNOT(aux[i * 2], aux[(i * 2) + 1], aux[i + ctls.size() / 2]);
  }
}

void CollectControls(cudaq::qview<> ctls, cudaq::qview<> aux,
                     int adjustment) __qpu__ {
  std::vector<std::reference_wrapper<cudaq::qubit>> castedCtls;
  for (auto &q : ctls)
    castedCtls.emplace_back(q);

  CollectControls(castedCtls, aux, adjustment);
}

/// When collecting controls, if there is an uneven number of original control
/// qubits then the last control and the second to last auxiliary will be
/// collected into the last auxiliary.
void AdjustForSingleControl(cudaq::qview<> ctls, cudaq::qview<> aux) __qpu__ {
  if (ctls.size() % 2 != 0)
    CCNOT(ctls[ctls.size() - 1], aux[ctls.size() - 3], aux[ctls.size() - 2]);
}

template <typename mod>
void x(const std::vector<std::reference_wrapper<cudaq::qubit>> &ctrls,
       cudaq::qubit &target) __qpu__ {
  static_assert(std::is_same_v<mod, cudaq::ctrl>);
  const std::size_t numCtrls = ctrls.size();
  if (numCtrls == 0) {
    x(target);
  } else if (numCtrls == 1) {
    cx(ctrls[0], target);
  } else if (numCtrls == 2) {
    CCNOT(ctrls[0], ctrls[1], target);
  } else {
    cudaq::qvector aux(numCtrls - 2);
    cudaq::compute_action(
        [&]() { CollectControls(ctrls, aux, 1 - (ctrls.size() % 2)); },
        [&]() {
          if (ctrls.size() % 2 != 0) {
            CCNOT(ctrls[ctrls.size() - 1], aux[ctrls.size() - 3], target);
          } else {
            CCNOT(aux[ctrls.size() - 3], aux[ctrls.size() - 4], target);
          }
        });
  }
}
} // namespace internal

template <typename mod, typename... QubitTy>
void x(cudaq::qubit &c0, cudaq::qubit &c1, QubitTy &...qubits) __qpu__ {
  static_assert(std::is_same_v<mod, cudaq::ctrl>);
  std::vector<std::reference_wrapper<cudaq::qubit>> ctls{{qubits...}};
  // Last qubit is the target
  ctls.pop_back();
  // Add the two explicit qubits
  ctls.emplace_back(c1);
  ctls.emplace_back(c0);
  internal::x<mod>(
      ctls, cudaq::getParameterPackVals<sizeof...(qubits) - 1>(qubits...));
}

template <typename mod>
void x(cudaq::qview<> ctrls, cudaq::qubit &target) __qpu__ {
  static_assert(std::is_same_v<mod, cudaq::ctrl>);
  std::vector<std::reference_wrapper<cudaq::qubit>> castedCtls;
  for (auto &q : ctrls)
    castedCtls.emplace_back(q);
  internal::x<mod>(castedCtls, target);
}
} // namespace cudaq
