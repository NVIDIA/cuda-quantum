/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "rt_host.h"

#include <cstring>

namespace cudaq::qclink {

namespace details {
static lqpu *logical_qpu_config = nullptr;
static std::unique_ptr<rt_host> qpu_rt_host;
static std::size_t kernelHandle = 0;
std::unordered_map<std::size_t, std::unique_ptr<compiled_kernel>> loaded;

void __set_lqpu_config(const lqpu *cfg) {
  details::logical_qpu_config = const_cast<lqpu *>(cfg);
}

template <typename T>
device_ptr device_ptr_from_ref(T &t) {
  return device_ptr(&t);
}
} // namespace details

void initialize(const lqpu *cfg);

device_ptr malloc(std::size_t size);
device_ptr malloc(std::size_t size, std::size_t devId);
template <typename T>
device_ptr malloc() {
  return malloc(sizeof(T));
}

/// @brief Allocate and set the data with given value. Return a device_ptr.
template <typename T>
device_ptr malloc_set(T t, std::optional<std::size_t> device = std::nullopt) {
  device_ptr ret;
  if (device.has_value())
    ret = malloc(sizeof(T), *device);
  else
    ret = qclink::malloc(sizeof(T));
  memcpy(ret, &t);
  return ret;
}

/// @brief Free the memory held by the given device_ptr.

void free(device_ptr &d);
// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src);
void memcpy(void *dest, const device_ptr &src);

/// @brief Copy the data from the given device_ptr to a host-side value.
template <typename T>
T memcpy(const device_ptr &src) {
  T t;
  memcpy(static_cast<void *>(&t), src);
  return t;
}

/// @brief Run any target-specific Quake compilation passes.
/// Returns a handle to the remotely JIT-ed code

handle load_kernel(const std::string &kernel_code,
                   const std::string &kernel_name);

void launch_kernel(handle kernelHandle, device_ptr &result,
                   const std::vector<device_ptr> &args);

void launch_kernel(handle kernelHandle, const std::vector<device_ptr> &args);

template <typename Ret, typename... Args>
Ret launch_kernel(handle kernelHandle, Args &&...args) {
  auto retPtr = qclink::malloc<Ret>();
  std::vector<device_ptr> argPtrs{
      details::device_ptr_from_ref<Args>(std::forward<Args>(args))...};
  launch_kernel(kernelHandle, retPtr, argPtrs);
  Ret ret;
  qclink::memcpy(&ret, retPtr);
  free(retPtr);
  return ret;
}

/// @brief shutdown the driver API. This should
/// kick of the disconnection of all channels.
void shutdown();

} // namespace cudaq::qclink

// Include all builtin extensions

#include "devices/all_devices.h"
#include "rt_hosts/all_rt_hosts.h"
