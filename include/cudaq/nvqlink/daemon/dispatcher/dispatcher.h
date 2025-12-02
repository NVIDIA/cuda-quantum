/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channel.h"
#include "cudaq/nvqlink/network/config.h"

#include <cstdint>
#include <memory>

namespace cudaq::nvqlink {

class ComputeConfig;
class FunctionRegistry;

/// Abstract dispatcher interface.
/// CPU and GPU modes have different implementations.
///
class Dispatcher {
public:
  virtual ~Dispatcher() = default;

  virtual void start() = 0;
  virtual void stop() = 0;

  virtual std::uint64_t get_packets_processed() const = 0;
  virtual std::uint64_t get_packets_sent() const = 0;
};

/// Factory
///
/// @param mode Datapath mode
/// @param backend Backend providing I/O and buffer management
/// @param registry Function registry
/// @param compute_config Compute config
/// @return Dispatcher
///
std::unique_ptr<Dispatcher>
create_dispatcher(DatapathMode mode, Channel *channel,
                  FunctionRegistry *registry,
                  const ComputeConfig &compute_config);

} // namespace cudaq::nvqlink
