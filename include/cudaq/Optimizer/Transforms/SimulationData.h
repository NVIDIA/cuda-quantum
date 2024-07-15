/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <numbers>
#include <vector>

// cudaq::state is defined in the runtime. The compiler will never need to know
// about its implementation and there should not be a circular build/library
// dependence because of it. Simply forward declare it, as it is notional.
namespace cudaq {
class state;
}


/// Owns the data
class SimulationData {
 public:
  typedef SimulationData (getSimulationDataFunc)(cudaq::state*);

  SimulationData(void *data, std::size_t size, std::size_t elementSize): 
    data(data), size(size), elementSize(elementSize) {}
  
  template <typename T> 
  std::vector<T> toVector() {
    assert(sizeof(T) == elementSize && "incorrect element size in simulation data");
    std::vector<T> result;

    for (auto i = 0; i < size; i++) {
      auto elePtr = reinterpret_cast<T*>(data + i*elementSize);
      result[i] = *elePtr;
    }

    return result;
  }

  ~SimulationData() {
    delete data;
  }

private:
  void* data;
  std::size_t size;
  std::size_t elementSize;
};


