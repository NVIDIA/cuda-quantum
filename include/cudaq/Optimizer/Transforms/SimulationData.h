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

#include <iostream>

// cudaq::state is defined in the runtime. The compiler will never need to know
// about its implementation and there should not be a circular build/library
// dependence because of it. Simply forward declare it, as it is notional.
namespace cudaq {
class state;
}


/// Owns the data
class SimulationStateData {
 public:
  typedef SimulationStateData (getDataFunc)(cudaq::state*);

  SimulationStateData(void *data, std::size_t size, std::size_t elementSize): 
    data(data), size(size), elementSize(elementSize) {}
  
  // template <typename T> 
  // std::vector<T> toVector() {
  //   assert(sizeof(T) == elementSize && "incorrect element size in simulation data");
  //   std::vector<T> result;

  //   std::cout << "SimulationStateData:" << std::endl;
  //   for (std::size_t i = 0; i < size; i++) {
  //     auto elePtr = reinterpret_cast<T*>(data) + i;
  //     result.push_back(*elePtr);
  //     std::cout << *elePtr << std::endl;
  //   }

  //   return result;
  // }

  ~SimulationStateData() {
    delete reinterpret_cast<int*>(data);
  }

  void* data;
  std::size_t size;
  std::size_t elementSize;
};


