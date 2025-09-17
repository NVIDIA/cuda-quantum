/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "device.h"
#include "lqpu.h"

#include "cudaq/nvqlink/rt_hosts/all_rt_hosts.h"

#include <cstring>

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
template <typename T>
device_ptr device_ptr_from_ref(T &t) {
  return device_ptr(&t);
}
} // namespace details

void initialize(const lqpu *cfg) {
  details::logical_qpu_config = const_cast<lqpu *>(cfg);
  details::qpu_rt_host = std::make_unique<any_rt_host>(
      nv_simulation_rt_host{*details::logical_qpu_config});
}

template <typename RTHostTy>
void initialize(const lqpu *cfg) {
  details::__set_lqpu_config(cfg);
  auto rtHost =
      std::make_unique<any_rt_host>(RTHostTy{*const_cast<lqpu *>(cfg)});
  details::__set_rt_host(std::move(rtHost));
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
  return std::visit(
      [size, devId](auto &&dev) -> device_ptr {
        using DeviceType = decltype(dev);
        if constexpr (has_data_marshalling_trait_v<DeviceType>) {
          return dev.malloc(size);
        } else {
          throw std::runtime_error(
              "device does not provide explicit data marshaling");
        }
      },
      device);
}

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
    ret = nvqlink::malloc(sizeof(T));
  memcpy(ret, &t);
  return ret;
}

/// @brief Free the memory held by the given device_ptr.

void free(device_ptr &d) {
  if (d.deviceId == std::numeric_limits<std::size_t>::max())
    return std::free(reinterpret_cast<void *>(d.handle));

  auto &device = details::logical_qpu_config->get_device(d.deviceId);
  return std::visit(
      [&d](auto &&dev) -> void {
        using DeviceType = decltype(dev);
        if constexpr (has_data_marshalling_trait_v<DeviceType>) {
          return dev.free(d);
        } else {
          throw std::runtime_error("no free method");
        }
      },
      device);
}
// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src) {
  if (arg.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(reinterpret_cast<void *>(arg.handle), src, arg.size);
    return;
  }

  auto &device = details::logical_qpu_config->get_device(arg.deviceId);
  return std::visit(
      [&arg, &src](auto &&dev) -> void {
        using DeviceType = decltype(dev);
        if constexpr (has_data_marshalling_trait_v<DeviceType>) {
          return dev.send(arg, src);
        } else {
          throw std::runtime_error("no send");
        }
      },
      device);
}

void memcpy(void *dest, const device_ptr &src) {
  if (src.deviceId == std::numeric_limits<std::size_t>::max()) {
    std::memcpy(dest, reinterpret_cast<void *>(src.handle), src.size);
    return;
  }

  auto &device = details::logical_qpu_config->get_device(src.deviceId);
  return std::visit(
      [&src, &dest](auto &&dev) {
        using DeviceType = decltype(dev);
        if constexpr (has_data_marshalling_trait_v<DeviceType>) {
          return dev.recv(dest, src);
        } else {
          throw std::runtime_error("no recv\n");
        }
      },
      device);
}

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

void launch_kernel(handle kernelHandle, const std::vector<device_ptr> &args) {
  device_ptr null;
  launch_kernel(kernelHandle, null, args);
}

template <typename Ret, typename... Args>
Ret launch_kernel(handle kernelHandle, Args &&...args) {
  auto retPtr = nvqlink::malloc<Ret>();
  std::vector<device_ptr> argPtrs{
      details::device_ptr_from_ref<Args>(std::forward<Args>(args))...};
  launch_kernel(kernelHandle, retPtr, argPtrs);
  Ret ret;
  nvqlink::memcpy(&ret, retPtr);
  free(retPtr);
  return ret;
}

/// @brief shutdown the driver API. This should
/// kick of the disconnection of all channels.
void shutdown() { details::logical_qpu_config = nullptr; }

} // namespace cudaq::nvqlink
