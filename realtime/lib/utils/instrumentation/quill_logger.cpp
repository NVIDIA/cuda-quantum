/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/utils/instrumentation/quill_logger.h"

namespace cudaq::nvqlink::quill_backend {

//===----------------------------------------------------------------------===//
// Domain Logger Instances
//===----------------------------------------------------------------------===//

// Define logger handles for each domain as global variables. This allows to
// avoid looking up loggers each time.

CustomLogger *logger_daemon = nullptr;
CustomLogger *logger_dispatcher = nullptr;
CustomLogger *logger_memory = nullptr;
CustomLogger *logger_channel = nullptr;
CustomLogger *logger_user = nullptr;
CustomLogger *logger_gpu = nullptr;

} // namespace cudaq::nvqlink::quill_backend

namespace cudaq::nvqlink::logger {

void initialize() {
  using namespace cudaq::nvqlink::quill_backend;

  //===--------------------------------------------------------------------===//
  // Start Quill Backend (dedicated logging thread)
  //===--------------------------------------------------------------------===//

  quill::BackendOptions backend_options;
  backend_options.thread_name = "nvqlink_log";

  // TODO: Evaluate whether we should pin the backend thread to specific CPU
  // backend_options.cpu_affinity = ..;

  quill::SignalHandlerOptions signal_handler_options;
  quill::Backend::start<CustomFrontendOptions>(backend_options,
                                               signal_handler_options);

  //===--------------------------------------------------------------------===//
  // Create Console Sink (matches current std::cout behavior)
  //===--------------------------------------------------------------------===//

  auto console_sink =
      CustomFrontend::create_or_get_sink<quill::ConsoleSink>("console_sink");

  // Optional: Configure sink pattern
  // Pattern format: [timestamp] [level] [logger_name] message
  // Default pattern is already good, but can be customized:
  // console_sink->set_pattern("[%(time)] [%(log_level)] [%(logger)]
  // %(message)");

  //===--------------------------------------------------------------------===//
  // Create Domain Loggers
  //===--------------------------------------------------------------------===//

  logger_daemon = CustomFrontend::create_or_get_logger("daemon", console_sink);
  logger_dispatcher =
      CustomFrontend::create_or_get_logger("dispatcher", console_sink);
  logger_memory = CustomFrontend::create_or_get_logger("memory", console_sink);
  logger_channel =
      CustomFrontend::create_or_get_logger("channel", console_sink);
  logger_user = CustomFrontend::create_or_get_logger("user", console_sink);
  logger_gpu = CustomFrontend::create_or_get_logger("gpu", console_sink);
}

void shutdown() {
  // Quill backend will automatically flush and stop on program exit, but we can
  // explicitly stop it here for clean shutdown.

  // Note: This is optional, Quill's destructor handles cleanup
}

} // namespace cudaq::nvqlink::logger
