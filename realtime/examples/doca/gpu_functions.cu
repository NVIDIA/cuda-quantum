/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// This file must be compiled with nvcc to properly handle device function
/// pointers. The main daemon example (.cpp) calls get_gpu_add_function_ptr()
/// to get the proper device function pointer.

#include "cudaq/nvqlink/network/serialization/gpu_input_stream.h"
#include "cudaq/nvqlink/network/serialization/gpu_output_stream.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

using namespace cudaq::nvqlink;

//===------------------------------------------------------------------------===
// GPU RPC Function: Add Two Numbers
//===------------------------------------------------------------------------===

/// @brief Simple GPU-callable RPC function for testing
///
/// This function demonstrates the NVQLink GPU RPC interface:
/// - Uses GPUInputStream to deserialize arguments
/// - Uses GPUOutputStream to serialize results
/// - Runs entirely on GPU with zero CPU involvement
///
__device__ std::int32_t gpu_add_numbers(GPUInputStream &in,
                                        GPUOutputStream &out) {
  // Read arguments from input stream
  std::int32_t a = in.read<std::int32_t>();
  std::int32_t b = in.read<std::int32_t>();

  // Compute result
  std::int32_t result = a + b;

  // Debug: print what we're computing
  printf("GPU: gpu_add_numbers called: %d + %d = %d\n", a, b, result);

  // Write result to output stream
  out.write(result);

  return 0; // Success
}

//===------------------------------------------------------------------------===
// Device Function Pointer Management
//===------------------------------------------------------------------------===

// Device function pointer type
using GPUFuncPtr = std::int32_t (*)(GPUInputStream &, GPUOutputStream &);

// Device variable holding the function pointer
// This is CRITICAL for proper CUDA function pointer semantics!
// Taking &gpu_add_numbers from host code gives a HOST address, not device
// address.
__device__ GPUFuncPtr d_gpu_add_numbers_ptr = gpu_add_numbers;

/// @brief Get the device function pointer from host code
///
/// This function uses cudaMemcpyFromSymbol to retrieve the actual device
/// function pointer stored in the __device__ variable.
///
/// @return Device function pointer suitable for use in GPU kernels
///
extern "C" void *get_gpu_add_function_ptr() {
  GPUFuncPtr host_ptr = nullptr;
  cudaError_t err = cudaMemcpyFromSymbol(&host_ptr, d_gpu_add_numbers_ptr,
                                         sizeof(GPUFuncPtr));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to get device function pointer: %s\n",
            cudaGetErrorString(err));
    return nullptr;
  }

  // Debug: print the retrieved pointer
  printf("[GPU_FUNC] Retrieved device function pointer: %p\n",
         (void *)host_ptr);

  return reinterpret_cast<void *>(host_ptr);
}
