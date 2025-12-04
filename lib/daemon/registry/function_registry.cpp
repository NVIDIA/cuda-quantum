/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/registry/function_registry.h"

#ifdef NVQLINK_HAVE_CUDA
#include <cuda_runtime.h>
#include "cudaq/nvqlink/daemon/registry/gpu_function_registry.h"
#endif

#include <stdexcept>

using namespace cudaq::nvqlink;

void FunctionRegistry::register_function(const FunctionMetadata &metadata) {
  auto it = functions_.find(metadata.function_id);
  if (it != functions_.end()) {
    // Hash collision detected - provide detailed error message
    throw std::runtime_error(
        "Hash collision detected: Function '" + metadata.name + "' (ID: 0x" +
        std::to_string(metadata.function_id) + ") collides with '" +
        it->second.name + "'. Consider renaming one of the functions.");
  }
  functions_[metadata.function_id] = metadata;
}

const FunctionMetadata *
FunctionRegistry::lookup(std::uint32_t function_id) const {
  auto it = functions_.find(function_id);
  return (it != functions_.end()) ? &it->second : nullptr;
}

#ifdef NVQLINK_HAVE_CUDA
FunctionRegistry::GPUFunctionTable
FunctionRegistry::get_gpu_function_table() const {
  // Allocate device memory for function table
  std::size_t count = functions_.size();

  void **host_func_ptrs = new void *[count];
  std::uint32_t *host_func_ids = new std::uint32_t[count];

  std::size_t idx = 0;
  for (const auto &[id, metadata] : functions_) {
    if (metadata.type == FunctionType::GPU) {
      host_func_ptrs[idx] = metadata.gpu_function;
      host_func_ids[idx] = id;
      ++idx;
    }
  }

  // Copy to device
  void **device_func_ptrs = nullptr;
  std::uint32_t *device_func_ids = nullptr;

  cudaMalloc(&device_func_ptrs, sizeof(void *) * count);
  cudaMalloc(&device_func_ids, sizeof(std::uint32_t) * count);

  cudaMemcpy(device_func_ptrs, host_func_ptrs, sizeof(void *) * count,
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_func_ids, host_func_ids, sizeof(std::uint32_t) * count,
             cudaMemcpyHostToDevice);

  delete[] host_func_ptrs;
  delete[] host_func_ids;

  gpu_table_.device_function_ptrs = device_func_ptrs;
  gpu_table_.function_ids = device_func_ids;
  gpu_table_.count = count;

  return gpu_table_;
}

void* FunctionRegistry::get_gpu_registry() const {
  // Build GPU function registry for NVQLink streaming interface
  // This converts from FunctionRegistry format to GPUFunctionRegistry format
  // Used by any GPU kernel using GPUInputStream/GPUOutputStream (e.g., DOCA kernel)
  
  if (device_gpu_registry_) {
    return device_gpu_registry_; // Return cached registry
  }
  
  // Count GPU functions
  std::size_t gpu_func_count = 0;
  for (const auto& [id, metadata] : functions_) {
    if (metadata.type == FunctionType::GPU) {
      ++gpu_func_count;
    }
  }
  
  if (gpu_func_count == 0) {
    return nullptr; // No GPU functions registered
  }
  
  if (gpu_func_count > GPUFunctionRegistry::MAX_FUNCTIONS) {
    throw std::runtime_error(
        "Too many GPU functions: " + std::to_string(gpu_func_count) +
        " exceeds maximum " + std::to_string(GPUFunctionRegistry::MAX_FUNCTIONS));
  }
  
  // Build host-side registry
  GPUFunctionRegistry host_registry;
  host_registry.num_functions = 0;
  
  for (const auto& [id, metadata] : functions_) {
    if (metadata.type == FunctionType::GPU) {
      GPUFunctionWrapper wrapper;
      wrapper.invoke = reinterpret_cast<GPUFunctionWrapper::InvokeFunc>(metadata.gpu_function);
      wrapper.function_id = id;
      
      host_registry.functions[host_registry.num_functions++] = wrapper;
    }
  }
  
  // Allocate device memory and copy
  GPUFunctionRegistry* device_registry = nullptr;
  cudaError_t err = cudaMalloc(&device_registry, sizeof(GPUFunctionRegistry));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device memory for GPU registry: " +
                           std::string(cudaGetErrorString(err)));
  }
  
  err = cudaMemcpy(device_registry, &host_registry, sizeof(GPUFunctionRegistry),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(device_registry);
    throw std::runtime_error("Failed to copy GPU registry to device: " +
                           std::string(cudaGetErrorString(err)));
  }
  
  device_gpu_registry_ = device_registry;
  return device_gpu_registry_;
}

#endif // NVQLINK_HAVE_CUDA
