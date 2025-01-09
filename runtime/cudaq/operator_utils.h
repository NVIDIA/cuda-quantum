/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "definition.h"
#include "matrix.h"
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cudaq {

inline std::map<std::string, std::complex<double>>
aggregate_parameters(const std::map<std::string, Definition> &param1,
                     const std::map<std::string, Definition> &param2) {
  std::map<std::string, std::complex<double>> merged_map = param1;

  for (const auto &[key, value] : param2) {
    /// FIXME: May just be able to remove this whole conditional block
    /// since we're not dealing with std::string entries, but instead
    /// complex doubles now.
    if (merged_map.find(key) != merged_map.end()) {
      // do nothing
    } else {
      merged_map[key] = value;
    }
  }

  return merged_map;
}
} // namespace cudaq