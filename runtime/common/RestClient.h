/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "nlohmann/json.hpp"
#include <map>
#include <string>

// Forward declarations to avoid including CPR header files
namespace cpr {
struct SslOptions;
}

namespace cudaq {

/// @brief The RestClient exposes a simple REST GET/POST
/// interface for interacting with remote REST servers.
class RestClient {
protected:
  // Use verbose printout
  bool verbose = false;

  /// SSL options to use for transfers
  std::unique_ptr<cpr::SslOptions> sslOptions;

public:
  /// @brief set verbose printout
  /// @param v
  void setVerbose(bool v) { verbose = v; }

  /// @brief Constructor
  RestClient();

  /// @brief Destructor
  ~RestClient();

  /// Post the message to the remote path at the provided URL.
  nlohmann::json post(const std::string_view remoteUrl,
                      const std::string_view path, nlohmann::json &postStr,
                      std::map<std::string, std::string> &headers,
                      bool enableLogging = true, bool enableSsl = false);
  /// Get the contents of the remote server at the given URL and path.
  nlohmann::json get(const std::string_view remoteUrl,
                     const std::string_view path,
                     std::map<std::string, std::string> &headers,
                     bool enableSsl = false);
  /// Put the message to the remote path at the provided URL.
  void put(const std::string_view remoteUrl, const std::string_view path,
           nlohmann::json &putData, std::map<std::string, std::string> &headers,
           bool enableLogging = true, bool enableSsl = false);
  /// Delete a resource at the provided URL.
  void del(const std::string_view remoteUrl, const std::string_view path,
           std::map<std::string, std::string> &headers,
           bool enableLogging = true, bool enableSsl = false);
  /// Download a resource at the provided URL and save it to the provided path.
  void download(const std::string_view remoteUrl, const std::string &filePath,
                bool enableLogging = true, bool enableSsl = false);
};
} // namespace cudaq
