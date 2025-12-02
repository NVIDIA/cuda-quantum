/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/registry/function_registry.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
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

#ifdef __CUDACC__
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
#endif
