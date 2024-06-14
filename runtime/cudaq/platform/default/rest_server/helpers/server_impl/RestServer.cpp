/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RestServer.h"
#include <cxxabi.h>

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
  // Disable streaming by setting stream threshold to 0 to enforce synchronous
  // TCP buffer write. Asynchronous buffer write (`asio::async_write`) is
  // susceptible to corruption if the app is shut down right after handling a
  // request.
  m_impl->app.stream_threshold(0);
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
    std::unordered_multimap<std::string, std::string> headers;
    for (const auto &[k, v] : req.headers)
      headers.emplace(k, v);

    return handler(req.body, headers).dump();
  } catch (std::exception &e) {
    const std::string errorMsg =
        std::string("Unhandled exception encountered: ") + e.what();
    return crow::response(500, errorMsg);
  } catch (...) {
    std::string exType = __cxxabiv1::__cxa_current_exception_type()->name();
    auto demangledPtr =
        __cxxabiv1::__cxa_demangle(exType.c_str(), nullptr, nullptr, nullptr);
    if (demangledPtr) {
      std::string demangledName(demangledPtr);
      const std::string errorMsg =
          "Unhandled exception of type " + demangledName;
      return crow::response(500, errorMsg);
    } else {
      return crow::response(500, "Unhandled exception of unknown type");
    }
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
