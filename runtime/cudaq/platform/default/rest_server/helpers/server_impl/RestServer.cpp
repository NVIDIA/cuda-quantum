/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RestServer.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "crow.h"
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

// Implement `cudaq::RestServer` interface using the CrowCpp library.
struct cudaq::RestServer::impl {
  crow::SimpleApp app;
};

cudaq::RestServer::RestServer(int port, const std::string &name) {
  m_impl = std::make_unique<impl>();
  m_impl->app.port(port);
  m_impl->app.server_name(name);
  m_impl->app.loglevel(crow::LogLevel::Warning);
  // Note: don't enable multi-threading (`m_impl->app.multithreaded()`) since
  // we're handling requests sequentially.
}
void cudaq::RestServer::start() { m_impl->app.run(); }
void cudaq::RestServer::stop() { m_impl->app.stop(); }
cudaq::RestServer::~RestServer() = default;

// Helper to invoke route handler: exceptions will be returned as 500 Internal
// Server Error.
static inline crow::response
invokeRouteHandler(const cudaq::RestServer::RouteHandler &handler,
                   const crow::request &req) {
  try {
    return handler(req.body).dump();
  } catch (std::exception &e) {
    const std::string errorMsg =
        std::string("Unhandled exception encountered: ") + e.what();
    return crow::response(500, errorMsg);
  }
}

void cudaq::RestServer::addRoute(Method routeMethod, const char *route,
                                 RouteHandler handler) {
  switch (routeMethod) {
  case (Method::GET):
    m_impl->app.route_dynamic(route).methods("GET"_method)(
        [handler](const crow::request &req) {
          return invokeRouteHandler(handler, req);
        });
    break;
  case (Method::POST):
    m_impl->app.route_dynamic(route).methods("POST"_method)(
        [handler](const crow::request &req) {
          return invokeRouteHandler(handler, (req));
        });
    break;
  }
}
