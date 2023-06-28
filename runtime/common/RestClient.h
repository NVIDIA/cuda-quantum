/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "nlohmann/json.hpp"
#include <map>
#include <string>

namespace cudaq {

/// @brief The RestClient exposes a simple REST GET/POST
/// interface for interacting with remote REST servers.
class RestClient {
protected:
  // Use verbose printout
  bool verbose = false;

public:
  /// @brief set verbose printout
  /// @param v
  void setVerbose(bool v) { verbose = v; }

  /// Post the message to the remote path at the provided URL.
  nlohmann::json post(const std::string_view remoteUrl,
                      const std::string_view path, nlohmann::json &postStr,
                      std::map<std::string, std::string> &headers);
  /// Get the contents of the remote server at the given url and path.
  nlohmann::json get(const std::string_view remoteUrl,
                     const std::string_view path,
                     std::map<std::string, std::string> &headers);

  ~RestClient() = default;
};
} // namespace cudaq
