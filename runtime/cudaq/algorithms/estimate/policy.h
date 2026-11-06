/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompileOptions.h"
#include "common/EstimateResult.h"
#include <functional>
#include <string>

namespace cudaq {

/// @brief Tag and options for resource estimation.
struct estimate_policy {
  static constexpr char name[] = "resource-count";
  using result_type = estimate_result;

  std::string kernelName;

  /// Choice function used to resolve measurements during the estimation.
  ///
  /// Invoked for every measurement to deterministically pick which branch to
  /// follow when the kernel branches on a measurement result.
  std::function<bool()> choice;

  /// Directory in which the runtime endpoint saves the compilation artifacts it
  /// produces for this estimation, like `nvq++ -save-temps`. Empty if the
  /// artifacts are not to be saved. Only endpoints that produce compilation
  /// artifacts of their own honor it.
  std::string saveTemps;

  friend CompileOptions get_compile_options_impl(const estimate_policy &);
};

} // namespace cudaq
