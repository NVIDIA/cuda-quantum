/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

namespace cudaq::qclink {

struct persistent_kernel_data {
  void *gpu_message_buffer;
  bool *trigger_flag;
  bool *completion_flag;
  cudaStream_t stream;
  void cleanup() {
    if (gpu_message_buffer)
      cudaFree(gpu_message_buffer);

    if (trigger_flag)
      cudaFree(trigger_flag);
    if (completion_flag)
      cudaFree(completion_flag);
  }
};

template <typename Derived>
class rdma_data {
public:
  rdma_data() = default;
  ~rdma_data() = default;
  
  void connect(persistent_kernel_data &data) {
    static_cast<Derived *>(this)->connect(data);
  }

  void disconnect(persistent_kernel_data &data) {
    static_cast<Derived *>(this)->disconnect(data);
  }

  void *get_raw_source() {
    return static_cast<Derived *>(this)->get_raw_source();
  }
  
  bool detected_buffer_change() {
    return static_cast<Derived *>(this)->detect_buffer_change();
  }
  
  void process_buffer_change(persistent_kernel_data &data) {
    static_cast<Derived *>(this)->process_buffer_change(data);
  }
  
};
} // namespace cudaq::qclink
