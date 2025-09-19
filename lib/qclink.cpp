/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qclink/qclink.h"
#include "utils/logger.h"

#include "cudaq/qclink/rt_host.h"

#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdarg.h>
#include <string.h>
#include <vector>

// FIXME This API assumes shmem rt_host_dispatch for now...

namespace cudaq::qclink {

void initialize(const lqpu *cfg) {
  details::logical_qpu_config = const_cast<lqpu *>(cfg);
  details::qpu_rt_host =
      rt_host::get("nv_simulation_rt_host", *const_cast<lqpu *>(cfg));
}

device_ptr malloc(std::size_t size) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  return {reinterpret_cast<uintptr_t>(std::malloc(size)), size};
}

device_ptr malloc(std::size_t size, std::size_t devId) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  auto &device = details::logical_qpu_config->get_device(devId);
  return device.as<explicit_data_marshalling_trait>()->malloc(size);
}

/// @brief Free the memory held by the given device_ptr.

void free(device_ptr &d) {
  if (d.deviceId == std::numeric_limits<std::size_t>::max())
    return std::free(reinterpret_cast<void *>(d.handle));
  auto &device = details::logical_qpu_config->get_device(d.deviceId);
  return device.as<explicit_data_marshalling_trait>()->free(d);
}

// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src) {
  if (arg.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(reinterpret_cast<void *>(arg.handle), src, arg.size);
    return;
  }
  auto &device = details::logical_qpu_config->get_device(arg.deviceId);
  return device.as<explicit_data_marshalling_trait>()->send(arg, src);
}

void memcpy(void *dest, const device_ptr &src) {
  if (src.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(dest, reinterpret_cast<void *>(src.handle), src.size);
    return;
  }
  auto &device = details::logical_qpu_config->get_device(src.deviceId);
  return device.as<explicit_data_marshalling_trait>()->recv(dest, src);
}

/// @brief Run any target-specific Quake compilation passes.
/// Returns a handle to the remotely JIT-ed code

handle load_kernel(const std::string &kernel_code,
                   const std::string &kernel_name) {
  auto compiled = details::qpu_rt_host->compile(kernel_code, kernel_name);
  details::loaded.insert({details::kernelHandle, std::move(compiled)});
  auto saved = details::kernelHandle;
  details::kernelHandle++;
  return saved;
}

void launch_kernel(handle kernelHandle, device_ptr &result,
                   const std::vector<device_ptr> &args) {
  // Launch it...
  auto iter = details::loaded.find(kernelHandle);
  if (iter == details::loaded.end())
    throw std::runtime_error("invalid kernel handle, cannot launch.");
  auto &progs = iter->second;
  details::qpu_rt_host->execute(*progs.get(), args, result);
}

void launch_kernel(handle kernelHandle, const std::vector<device_ptr> &args) {
  device_ptr null;
  launch_kernel(kernelHandle, null, args);
}

/// @brief shutdown the driver API. This should
/// kick of the disconnection of all channels.
void shutdown() {
  details::logical_qpu_config = nullptr;
  details::qpu_rt_host.reset();
  device_counter = 0;
}


} // namespace cudaq::qclink
