
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qreg.h"
#include <vector>

namespace cudaq {
/// @brief The `plus` gate
// U|0> -> |1>, U|1> -> |2>, ..., and U|d> -> |0>
template <std::size_t T>
void plus(cudaq::qudit<T> &q) {
  cudaq::getExecutionManager()->apply("plusGate", {}, {},
                                      {{q.n_levels(), q.id()}});
}

/// @brief The `phase shift` gate
template <std::size_t T>
void phase_shift(cudaq::qudit<T> &q, const double &phi) {
  cudaq::getExecutionManager()->apply("phaseShiftGate", {phi}, {},
                                      {{q.n_levels(), q.id()}});
}

/// @brief The `beam splitter` gate
template <std::size_t T>
void beam_splitter(cudaq::qudit<T> &q, cudaq::qudit<T> &r,
                   const double &theta) {
  cudaq::getExecutionManager()->apply(
      "beamSplitterGate", {theta}, {},
      {{q.n_levels(), q.id()}, {r.n_levels(), r.id()}});
}

/// @brief Measure a qudit
template <std::size_t T>
int mz(cudaq::qudit<T> &q) {
  return cudaq::getExecutionManager()->measure({q.n_levels(), q.id()});
}

/// @brief Measure a vector of qudits
template <std::size_t T>
std::vector<int> mz(cudaq::qreg<cudaq::dyn, T> &q) {
  std::vector<int> ret;
  for (auto &qq : q)
    ret.emplace_back(mz(qq));
  return ret;
}
} // namespace cudaq