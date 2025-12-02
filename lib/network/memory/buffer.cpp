/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/memory/buffer.h"

#include <stdexcept>

using namespace cudaq::nvqlink;

Buffer::Buffer(void *base_addr, std::size_t total_size, std::size_t headroom,
               std::size_t tailroom)
    : base_addr_(base_addr), total_size_(total_size), headroom_(headroom),
      tailroom_(tailroom), data_len_(0) {

  // Data starts after headroom
  data_ptr_ = static_cast<char *>(base_addr_) + headroom_;
}

void *Buffer::prepend(std::size_t bytes) {
  if (bytes > headroom_) {
    throw std::runtime_error("Insufficient headroom for prepend");
  }

  data_ptr_ = static_cast<char *>(data_ptr_) - bytes;
  data_len_ += bytes;
  headroom_ -= bytes;

  return data_ptr_;
}
