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

namespace cudaq {
/// Generic REST server interface
class RestServer {
private:
  // Forward declare implementation details.
  struct impl;
  std::unique_ptr<impl> m_impl;

public:
  // Signature of endpoint handler: accepting a string (HTTP request payload)
  // and returning a JSON object.
  using RouteHandler = std::function<nlohmann::json(
      const std::string &,
      const std::unordered_multimap<std::string, std::string> &)>;
  enum class Method { GET, POST };
  // Create a REST server serving at a specific port.
  RestServer(int port, const std::string &name = "cudaq");
  // Add a route (endpoint) handler.
  void addRoute(Method routeMethod, const char *route, RouteHandler handler);
  // Start the server.
  void start();
  // Stop the server.
  void stop();
  // Destructor
  ~RestServer();
};
} // namespace cudaq
