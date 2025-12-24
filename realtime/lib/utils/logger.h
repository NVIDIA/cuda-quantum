/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// spdlog's bundled fmtlib
#include <spdlog/fmt/fmt.h>

#include <functional>
#include <mutex>
#include <string>

namespace cudaq {

class Logger {
public:
  enum class Level { Info, Debug, Warn };

  // Sets the actual log handler. Call once at startup, passing a function that
  // takes a message string and a log level. You may call multiple times.
  static void
  set_handler(std::function<void(Level, const std::string &)> handler) {
    std::lock_guard<std::mutex> lock(handler_mutex());
    handler_ref() = std::move(handler);
  }

  // Logging methods with fmt-style formatting.
  template <typename... Args>
  static void log(fmt::format_string<Args...> fmt_str, Args &&...args) {
    log_dispatch(Level::Info,
                 fmt::format(fmt_str, std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void debug(fmt::format_string<Args...> fmt_str, Args &&...args) {
    log_dispatch(Level::Debug,
                 fmt::format(fmt_str, std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void warn(fmt::format_string<Args...> fmt_str, Args &&...args) {
    log_dispatch(Level::Warn,
                 fmt::format(fmt_str, std::forward<Args>(args)...));
  }

private:
  // Mutex and handler storage (function-static to avoid ODR/use issues)
  static std::mutex &handler_mutex() {
    static std::mutex mtx;
    return mtx;
  }
  static std::function<void(Level, const std::string &)> &handler_ref() {
    static std::function<void(Level, const std::string &)> handler{};
    return handler;
  }

  static void log_dispatch(Level lvl, const std::string &msg) {
    std::lock_guard<std::mutex> lock(handler_mutex());
    if (handler_ref())
      handler_ref()(lvl, msg);
  }
};

} // namespace cudaq
