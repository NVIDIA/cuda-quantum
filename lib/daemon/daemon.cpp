/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/daemon.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <stdexcept>

using namespace cudaq::nvqlink;

Daemon::Daemon(const DaemonConfig &config, std::unique_ptr<Channel> channel)
    : config_(config), channel_(std::move(channel)) {
  nvqlink::logger::initialize();
  nvqlink::profiler::initialize();
  NVQLINK_TRACE_FULL(DOMAIN_DAEMON, "ctor");

  if (!channel_)
    throw std::invalid_argument("Daemon: channel cannot be null");

  validate_config();
  function_registry_ = std::make_unique<FunctionRegistry>();

  NVQLINK_LOG_INFO(DOMAIN_DAEMON, "Daemon {} initialized with provided channel",
                   config_.id);
}

Daemon::~Daemon() {
  NVQLINK_TRACE_FULL(DOMAIN_DAEMON, "dtor");
  if (running_)
    stop();

  nvqlink::profiler::shutdown();
  nvqlink::logger::shutdown();
}

void Daemon::start() {
  NVQLINK_TRACE_FULL(DOMAIN_DAEMON, "start");
  if (running_.exchange(true)) {
    NVQLINK_LOG_INFO(DOMAIN_DAEMON, "Daemon already running");
    return;
  }

  initialize_dispatcher();
  dispatcher_->start();

  NVQLINK_LOG_INFO(DOMAIN_DAEMON, "Daemon {} started successfully", config_.id);
}

void Daemon::stop() {
  NVQLINK_TRACE_FULL(DOMAIN_DAEMON, "stop");
  if (!running_.exchange(false))
    return;

  NVQLINK_LOG_INFO(DOMAIN_DAEMON, "Stopping daemon {}...", config_.id);

  if (dispatcher_)
    dispatcher_->stop();

  NVQLINK_LOG_INFO(DOMAIN_DAEMON, "Daemon {} stopped", config_.id);
}

void Daemon::register_function(const FunctionMetadata &metadata) {
  if (running_)
    throw std::runtime_error(
        "Cannot register functions while daemon is running");

  if (!function_registry_)
    function_registry_ = std::make_unique<FunctionRegistry>();

  function_registry_->register_function(metadata);
}

Daemon::Stats Daemon::get_stats() const {
  Stats stats = stats_;
  if (dispatcher_) {
    stats.packets_received = dispatcher_->get_packets_processed();
    stats.packets_sent = dispatcher_->get_packets_sent();
  }
  return stats;
}

void Daemon::validate_config() const {
  // Validate datapath-backend compatibility
  if (config_.datapath_mode == DatapathMode::GPU) {
    // TODO: Somewhere, probably not here, we need to validate whether the
    // backend supports GPU datapath.
    if (!config_.compute.gpu_device_id.has_value())
      throw std::invalid_argument(
          "GPU datapath requires GPU device configuration");
  }

  if (config_.datapath_mode == DatapathMode::CPU) {
    if (config_.compute.cpu_cores.empty())
      throw std::invalid_argument(
          "CPU datapath requires CPU core configuration");
  }
}

void Daemon::initialize_dispatcher() {
  dispatcher_ = create_dispatcher(config_.datapath_mode, channel_.get(),
                                  function_registry_.get(), config_.compute);
  dispatcher_->start();
}
