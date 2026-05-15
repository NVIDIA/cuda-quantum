/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq_internal::device_call {

// Internal plugin boundary used by cudaq::realtime.  The public realtime API
// lives in the core runtime and resolves this interface from the device_call
// runtime library only when realtime support is enabled.
class DeviceCallRuntimePlugin {
public:
  virtual ~DeviceCallRuntimePlugin() = default;

  // Configure the device_call runtime from cudaq::realtime::initialize.
  virtual void initialize(int argc, char **argv) = 0;
  // Tear down device_call runtime: e.g., any service data created by
  // initialize.
  virtual void finalize() = 0;
};

} // namespace cudaq_internal::device_call
