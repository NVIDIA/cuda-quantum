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
  /// @brief The source code of the objective function and its call as a string.
  std::string code;

  /// @brief The local namespace of the objective function as a json object.
  json locals;

  /// @brief The global namespace of the objective function as a json object.
  json globals;

  /// @brief A computed optimal value
  std::optional<double> optimalValue = std::nullopt;

  /// @brief The optimal parameters returned on execution.
  std::vector<double> optimalParameters;

  SerializedCodeExecutionContext() = default;
  SerializedCodeExecutionContext(std::string c, json l, json g)
      : code(std::move(c)), locals(std::move(l)), globals(std::move(g)) {}
  ~SerializedCodeExecutionContext() = default;

  // Serialization
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SerializedCodeExecutionContext, code, locals,
                                 globals);
};
} // namespace cudaq
