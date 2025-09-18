/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma nv_diag_suppress = unsigned_compare_with_zero
#pragma nv_diag_suppress = unrecognized_gcc_pragma
#pragma nv_diag_suppress = 128
#pragma nv_diag_suppress = 2417

#include "cudaq/nvqlink/devices/cuda_device.h"

#include "../utils/logger.h"

#include <cstring>
#include <fstream>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      const char *errName;                                                     \
      cuGetErrorName(result, &errName);                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s failed with error: %s\n",       \
              __FILE__, __LINE__, #call, errName);                             \
    }                                                                          \
  } while (0)

namespace cudaq::nvqlink {

/// Array of loaded CUDA modules.
CUmodule *loadedModules = nullptr;

/// Map of loaded callback (kernel) function names to CUfunction pointers.
std::map<std::string, CUfunction *> loadedCallbacks;

/// CUDA context for this channel.
CUcontext context;

/// CUDA device handle.
CUdevice cudaGPUDevice;

std::size_t to_handle(void *ptr) { return reinterpret_cast<uintptr_t>(ptr); }

template <typename Applicator>
auto runOnCorrectDevice(std::size_t cudaDevice, const Applicator &applicator)
    -> std::invoke_result_t<Applicator> {
  int dev;
  cudaGetDevice(&dev);
  if (cudaDevice == dev)
    return applicator();

  cudaSetDevice(cudaDevice);
  if constexpr (std::is_void_v<std::invoke_result_t<Applicator>>) {
    applicator();
    cudaSetDevice(dev);
    return;
  } else {
    auto val = applicator();
    cudaSetDevice(dev);
    return val;
  }
}

void cuda_device::connect() {
  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cudaGPUDevice, cudaDevice));
  CUDA_CHECK(cuCtxCreate(&context, 0, cudaGPUDevice));
  loadedModules = new CUmodule[device_callbacks.size()];
  std::size_t i = 0;
  for (auto &[availFatBin, devFuncs] : device_callbacks) {
    cudaq::Logger::log("Loading available fatbin file - {}", availFatBin);
    std::ifstream file(availFatBin, std::ios::binary);
    std::vector<char> fatbin_data((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    auto &mod = loadedModules[i++];
    // Load from memory buffer
    CUDA_CHECK(
        cuModuleLoadDataEx(&mod, fatbin_data.data(), 0, nullptr, nullptr));

    for (auto &devFunc : devFuncs) {
      cudaq::Logger::log("\twithin fatbin, loading {}", devFunc.name);
      CUfunction *function = new CUfunction;
      // Load the kernel function
      CUDA_CHECK(cuModuleGetFunction(function, mod, devFunc.name.c_str()));
      loadedCallbacks.insert({devFunc.name, function});
    }
  }
}

void cuda_device::disconnect() {
  for (std::size_t i = 0; i < device_callbacks.size(); i++)
    CUDA_CHECK(cuModuleUnload(loadedModules[i]));

  delete loadedModules;

  CUDA_CHECK(cuCtxDestroy(context));
}

void *cuda_device::resolve_pointer(device_ptr &devPtr) {
  return local_memory_pool.at(devPtr.handle);
}

device_ptr cuda_device::malloc(std::size_t size) {
  return runOnCorrectDevice(cudaDevice, [&]() -> device_ptr {
    void *ptr = nullptr;
    cudaMalloc(&ptr, size);
    cudaMemset(ptr, 0, size);
    device_ptr devPtr{to_handle(ptr), size, get_id()};
    local_memory_pool.insert({devPtr.handle, ptr});
    cudaq::Logger::log(
        "cuda channel (device {}) allocating data of size {}, hdl {}.",
        get_id(), size, devPtr.handle);
    return devPtr;
  });
}

void cuda_device::free(device_ptr &d) {
  cudaq::Logger::log("cuda channel freeing data.");
  runOnCorrectDevice(cudaDevice,
                     [&]() { cudaFree(local_memory_pool.at(d.handle)); });
}

void cuda_device::send(device_ptr &src, const void *dst) {
  cudaq::Logger::log("cuda channel copying data to GPU.");
  runOnCorrectDevice(cudaDevice, [&]() {
    cudaMemcpy(local_memory_pool.at(src.handle), dst, src.size,
               cudaMemcpyHostToDevice);
  });
}

void cuda_device::recv(void *dest, const device_ptr &src) {
  cudaq::Logger::log("cuda channel copying data from GPU {}.", src.handle);
  runOnCorrectDevice(cudaDevice, [&]() {
    cudaMemcpy(dest, local_memory_pool.at(src.handle), src.size,
               cudaMemcpyDeviceToHost);
  });
}

void cuda_device::launch_callback(const std::string &funcName,
                                  device_ptr &result,
                                  const std::vector<device_ptr> &args) {
  if (!result.is_nullptr())
    throw std::runtime_error("cuda_channel does not provide return values.");

  cudaq::Logger::log("launching callback {}.", funcName);
  auto *cuFunc = loadedCallbacks.at(funcName);
  std::vector<void *> argsVec;
  std::transform(args.begin(), args.end(), std::back_inserter(argsVec),
                 [&](device_ptr arg) -> void * {
                   if (arg.is_host_value())
                     return arg.host_value;

                   return &local_memory_pool.at(arg.handle);
                 });
  CUDA_CHECK(
      cuLaunchKernel(*cuFunc, 1024, 1, 1, 1, 1, 1, 0, 0, argsVec.data(), 0));
}

} // namespace cudaq::nvqlink
