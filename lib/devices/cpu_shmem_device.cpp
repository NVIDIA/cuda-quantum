/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qclink/devices/cpu_shmem_device.h"

#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <variant>

namespace cudaq::qclink {

/// @brief Convert a shared memory device pointer handle to its raw pointer.
/// @param d Device pointer handle.
/// @return Raw pointer representation of the device memory.
inline void *to_ptr(const device_ptr &d) {
  return reinterpret_cast<void *>(d.handle);
}

/// @brief Convert a raw pointer to its handle representation.
/// @param ptr Raw pointer.
/// @return Handle representation of the pointer.
inline std::size_t to_handle(void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr);
}

void cpu_shmem_device::connect() {

  // In the future, we may want to loop through
  // the libraries and see if there is a symbol for a
  // registered unmarshaller function that MLIR generates

  for (auto &[availLib, devFuncs] : device_callbacks) {
    // Open the library
    auto *hdl = dlopen(availLib.c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (!hdl)
      throw std::runtime_error("could not load " + availLib);
    handles.push_back(hdl);

    for (auto &devFunc : devFuncs) {
      if (!devFunc.unmarshaller.has_value())
        continue;

      // Store the symbol
      void *sym = dlsym(hdl, devFunc.name.c_str());
      whatIsThisCalled.insert({devFunc.name, {sym, devFunc}});
    }
  }
}
void cpu_shmem_device::disconnect() {
  for (auto *hdl : handles) {
    dlclose(hdl);
  }
}

void *cpu_shmem_device::resolve_pointer(device_ptr &devPtr) {
  return reinterpret_cast<void *>(devPtr.handle);
}

device_ptr cpu_shmem_device::malloc(std::size_t size) {
  return {to_handle(std::malloc(size)), size, get_id()};
}

void cpu_shmem_device::free(device_ptr &d) { std::free(to_ptr(d)); }

void cpu_shmem_device::send(device_ptr &dest, const void *src) {
  std::memcpy(to_ptr(dest), src, dest.size);
}

void cpu_shmem_device::recv(void *dest, const device_ptr &src) {
  std::memcpy(dest, to_ptr(src), src.size);
}

void cpu_shmem_device::launch_callback(const std::string &funcName,
                                       device_ptr &result,
                                       const std::vector<device_ptr> &args) {
  auto iter = whatIsThisCalled.find(funcName);
  // In the future we may also have automated MLIR code
  if (iter == whatIsThisCalled.end())
    throw std::runtime_error("invalid callback requested");

  auto *sym = iter->second.first;
  auto unmarshaller = iter->second.second.unmarshaller.value();
  unmarshaller(sym, result, args);
  return;
}

} // namespace cudaq::qclink
