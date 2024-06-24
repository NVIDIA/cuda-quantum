/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nlohmann/json.hpp"
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace cudaq {

/// @brief The SerializedCodeExecutionContext is an abstraction to indicate
/// how a serialized code should be executed.
class SerializedCodeExecutionContext {
public:
  /// @brief All variables visible to the Python \p source_code to execute, as a
  /// JSON-like string object.
  std::string scoped_var_dict;

  /// @brief The source code of the objective function and its call as a string.
  std::string source_code;

  SerializedCodeExecutionContext() = default;
  ~SerializedCodeExecutionContext() = default;

  // Serialization
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SerializedCodeExecutionContext,
                                 scoped_var_dict, source_code);
};
} // namespace cudaq
