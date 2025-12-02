/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace cudaq::nvqlink {

/// CPU function signature: type-safe streaming interface.
///
/// User functions receive an InputStream for reading arguments and
/// an OutputStream for writing results. Both streams operate on the
/// same underlying buffer (zero-copy), but provide separate, type-safe
/// interfaces for reading and writing.
///
/// @param in Input stream for reading function arguments
/// @param out Output stream for writing function results
/// @return Status code (0 = success, non-zero = error)
///
/// @example
/// ```cpp
/// int add_function(InputStream& in, OutputStream& out) {
///   int a = in.read<int>();
///   int b = in.read<int>();
///   out.write(a + b);
///   return 0;
/// }
/// ```
using CPUFunction = std::function<int(InputStream &in, OutputStream &out)>;

// GPU function signature (device function pointer)
using GPUFunction = void *; // Actually: __device__ function pointer

enum class FunctionType { CPU, GPU };

struct FunctionMetadata {
  std::uint32_t function_id;
  std::string name;
  FunctionType type;
  std::size_t max_result_size;

  // CPU mode
  CPUFunction cpu_function;

  // GPU mode
  GPUFunction gpu_function;
};

/// Registry for user-defined RPC functions.
/// Functions must be registered before daemon starts.
///
class FunctionRegistry {
public:
  FunctionRegistry() = default;

  void register_function(const FunctionMetadata &metadata);

  const FunctionMetadata *lookup(std::uint32_t function_id) const;

  //===--------------------------------------------------------------------===//
  // GPU mode
  //===--------------------------------------------------------------------===//

  // For GPU mode: get device-side function table
  struct GPUFunctionTable {
    void **device_function_ptrs; // Array of device function pointers
    std::uint32_t *function_ids; // Corresponding function IDs
    std::size_t count;
  };
  GPUFunctionTable get_gpu_function_table() const;

private:
  std::unordered_map<std::uint32_t, FunctionMetadata> functions_;
  mutable GPUFunctionTable gpu_table_; // Cached GPU function table
};

} // namespace cudaq::nvqlink
