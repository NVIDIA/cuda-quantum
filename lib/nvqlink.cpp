/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/nvqlink.h"
#include "utils/logger.h"

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

namespace cudaq::nvqlink {

namespace details {
static lqpu *logical_qpu_config = nullptr;
static std::unique_ptr<any_rt_host> qpu_rt_host;
static std::size_t kernelHandle = 0;
std::unordered_map<std::size_t, std::unique_ptr<compiled_kernel>> loaded;

void __set_lqpu_config(const lqpu *cfg) {
  details::logical_qpu_config = const_cast<lqpu *>(cfg);
}
void __set_rt_host(std::unique_ptr<any_rt_host> &&rt) {
  details::qpu_rt_host = std::move(rt);
}
} // namespace details

void initialize(const lqpu *cfg) {
  details::logical_qpu_config = const_cast<lqpu *>(cfg);
  details::qpu_rt_host = std::make_unique<any_rt_host>(
      nv_simulation_rt_host{*details::logical_qpu_config});
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
  auto channel = std::visit([](auto &&d) { return d.get_channel(); }, device);
  return std::visit([size, devId](auto &&ch) { return ch.malloc(size); },
                    channel);
}

void free(device_ptr &d) {
  if (d.deviceId == std::numeric_limits<std::size_t>::max())
    return std::free(reinterpret_cast<void *>(d.handle));

  auto &device = details::logical_qpu_config->get_device(d.deviceId);
  auto channel = std::visit([](auto &&d) { return d.get_channel(); }, device);
  return std::visit([&d](auto &&ch) { return ch.free(d); }, channel);
}

// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src) {
  if (arg.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(reinterpret_cast<void *>(arg.handle), src, arg.size);
    return;
  }

  auto &device = details::logical_qpu_config->get_device(arg.deviceId);
  auto channel = std::visit([](auto &&d) { return d.get_channel(); }, device);
  return std::visit([&arg, &src](auto &&ch) { return ch.send(arg, src); },
                    channel);
}

void memcpy(void *dest, const device_ptr &src) {
  if (src.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(dest, reinterpret_cast<void *>(src.handle), src.size);
    return;
  }

  auto &device = details::logical_qpu_config->get_device(src.deviceId);
  auto channel = std::visit([](auto &&d) { return d.get_channel(); }, device);
  return std::visit([&src, &dest](auto &&ch) { return ch.recv(dest, src); },
                    channel);
}

handle load_kernel(const std::string &kernel_code,
                   const std::string &kernel_name) {
  auto compiled = std::visit(
      [&](auto &&rt_host) { return rt_host.compile(kernel_code, kernel_name); },
      *details::qpu_rt_host);

  details::loaded.insert({details::kernelHandle, std::move(compiled)});

  // Increment the handle and return
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

  // Execute the ,ernelprogram
  auto &progs = iter->second;
  std::visit(
      [&](auto &&rt_host) { rt_host.execute(*progs.get(), args, result); },
      *details::qpu_rt_host);
}

void shutdown() { details::logical_qpu_config = nullptr; }

} // namespace cudaq::nvqlink
