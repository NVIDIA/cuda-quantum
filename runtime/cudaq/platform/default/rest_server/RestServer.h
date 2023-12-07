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
class RestServer {
private:
  struct impl;
  std::unique_ptr<impl> m_impl;

public:
  using RouteHandler = std::function<nlohmann::json(const std::string &)>;
  enum class Method { GET, POST };
  RestServer(int port, const std::string &name = "cudaq");
  void start();
  void stop();
  ~RestServer();
  void addRoute(Method routeMethod, const char *route, RouteHandler handler);
};
} // namespace cudaq
