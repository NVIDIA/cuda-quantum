/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <numbers>
#include <tuple>
#include <vector>

namespace cudaq::opt {

/// Stores argument values.
/// Stores data, element size, and number of elements for each argument.
/// If the data pointer is owned, cleans up the data on destruction.
class ArgumentDataStore {
  using DataDeleter = std::function<void(void *)>;

  std::vector<std::tuple<void *, std::size_t, std::size_t>> states{};
  std::vector<DataDeleter> cleanup{};

public:
  ArgumentDataStore() = default;

  template <typename T>
  void addData(T *data, std::size_t size, std::size_t elementSize,
               DataDeleter deleter) {
    states.push_back({data, size, elementSize});
    cleanup.push_back(deleter);
  }

  std::tuple<void *, std::size_t, std::size_t>
  getData(std::size_t index) const {
    return states[index];
  }

  bool isEmpty() const noexcept { return states.empty(); }

  ~ArgumentDataStore() {
    for (std::size_t i = 0; i < states.size(); i++) {
      auto [data, size, elementSize] = states[i];
      cleanup[i](data);
    }
    states.clear();
    cleanup.clear();
  }
};
} // namespace cudaq::opt
