/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nlohmann/json_fwd.hpp"
#include <memory>

namespace cudaq {

/// @brief Opaque `pimpl` wrapper around `nlohmann::json`.
///
/// Use this type in headers to avoid pulling in the heavy nlohmann/json.hpp.
/// Only cudaq_json.cpp includes the full `nlohmann` header. Callers that need
/// to operate on the underlying JSON value should call get() in a `.cpp` file
/// that includes nlohmann/json.hpp directly.
class cudaq_json {
public:
  /// @brief Default-construct an empty JSON object.
  cudaq_json();
  /// @brief Copy constructor.
  cudaq_json(const cudaq_json &);
  /// @brief Move constructor.
  cudaq_json(cudaq_json &&) noexcept;
  /// @brief Copy-assignment operator.
  cudaq_json &operator=(const cudaq_json &);
  /// @brief Move-assignment operator.
  cudaq_json &operator=(cudaq_json &&) noexcept;
  ~cudaq_json();

  /// @brief Construct from an existing nlohmann::json value (copies).
  cudaq_json(const nlohmann::json &);
  /// @brief Construct from an rvalue nlohmann::json (moves).
  cudaq_json(nlohmann::json &&);

  /// @brief Access the underlying nlohmann::json.
  nlohmann::json &get();
  /// @brief Access the underlying nlohmann::json (`const`).
  const nlohmann::json &get() const;

  /// @brief `Dereference` to the underlying nlohmann::json.
  nlohmann::json &operator*();
  /// @brief `Dereference` to the underlying nlohmann::json (`const`).
  const nlohmann::json &operator*() const;
  /// @brief Arrow access to the underlying nlohmann::json.
  nlohmann::json *operator->();
  /// @brief Arrow access to the underlying nlohmann::json (`const`).
  const nlohmann::json *operator->() const;

private:
  std::unique_ptr<nlohmann::json> impl;
};

} // namespace cudaq
