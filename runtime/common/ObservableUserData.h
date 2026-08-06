/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Helpers for attaching a full spin_op to KernelExecution::user_data when a
/// REST / custom QPU backend evaluates observables server-side (e.g. Fermioniq
/// or an external plugin that registers its own QPU).

#pragma once

#include "common/KernelExecution.h"
#include "nlohmann/json.hpp"
#include "cudaq/runtime/logger/cudaq_fmt.h"
#include "cudaq/spin_op.h"
#include <cmath>

namespace cudaq {

/// @brief Attach a full spin observable to a KernelExecution as
/// `user_data["observable"]` for server-side observe backends.
///
/// Format matches Fermioniq / external REST plugins:
/// `[["Z0", "0.5+0.0j"], ["Z0 Z1", "0.3+0.0j"], ...]`
inline void attachObservableUserData(KernelExecution &code,
                                     const spin_op &spin) {
  nlohmann::json user_data = nlohmann::json::object();
  nlohmann::json obs = nlohmann::json::array();
  for (const auto &term : spin) {
    nlohmann::json terms = nlohmann::json::array();
    terms.push_back(term.get_term_id());
    auto coeff = term.evaluate_coefficient();
    auto coeff_str = cudaq_fmt::format("{}{}{}j", coeff.real(),
                                       coeff.imag() < 0.0 ? "-" : "+",
                                       std::fabs(coeff.imag()));
    terms.push_back(coeff_str);
    obs.push_back(std::move(terms));
  }
  user_data["observable"] = std::move(obs);
  code.user_data = cudaq_json(std::move(user_data));
}

} // namespace cudaq
