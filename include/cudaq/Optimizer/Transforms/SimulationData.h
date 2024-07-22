/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <numbers>
#include <vector>

/// Used to collect the simulation state data for a set of `cudaq::state`s.
/// Assumes the data pointers stored are copied into a new memory, and takes
/// ownership of that memory.
class SimulationStateDataStore {
  std::vector<std::tuple<void*, std::size_t>> states{};
  std::size_t _elementSize;

public:
  SimulationStateDataStore(std::size_t eSize = 0): _elementSize(eSize) {};

  void setElementSize(std::size_t eSize) {
    assert((_elementSize == 0 || _elementSize == eSize) && "Conflicting simulation data sizes in one collection");
    _elementSize = eSize;
  }

  std::size_t getElementSize() const {
    return _elementSize;
  }

  void addData(void *data, std::size_t size) {
    states.push_back({data, size});
  }

  std::tuple<void*, std::size_t> getData(std::size_t index) const {
    return states[index];
  }

  ~SimulationStateDataStore() { 
    for (auto [data, size]: states) 
      delete reinterpret_cast<int *>(data);
  }
};
