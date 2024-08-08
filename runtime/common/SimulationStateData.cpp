/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/SimulationStateData.h"
#include "common/SimulationState.h"
#include "cudaq/Optimizer/Transforms/ArgumentDataStore.h"
#include "cudaq/qis/state.h"

#include <cstdlib>

namespace cudaq::runtime {

cudaq::opt::ArgumentDataStore readSimulationStateData(
    std::pair<std::size_t, std::vector<std::size_t>> &argumentLayout,
    const void *args) {
  cudaq::opt::ArgumentDataStore dataStore;

  auto offsets = argumentLayout.second;
  for (std::size_t argNum = 0; argNum < offsets.size(); argNum++) {
    auto offset = offsets[argNum];

    cudaq::state *state;
    std::memcpy(&state, ((const char *)args) + offset, sizeof(cudaq::state *));

    auto precision = state->get_precision();
    auto stateVector = state->get_tensor();
    auto numElements = stateVector.get_num_elements();
    auto elementSize = stateVector.element_size();

    if (state->is_on_gpu()) {
      if (precision == cudaq::SimulationState::precision::fp32) {
        assert(elementSize == sizeof(std::complex<float>) &&
               "Incorrect complex<float> element size");
        auto *hostData = new std::complex<float>[numElements];
        state->to_host(hostData, numElements);
        dataStore.addData(hostData, numElements, elementSize, [](void *ptr) {
          delete static_cast<std::complex<float> *>(ptr);
        });
      } else {
        assert(elementSize == sizeof(std::complex<double>) &&
               "Incorrect complex<double> element size");
        auto *hostData = new std::complex<double>[numElements];
        state->to_host(hostData, numElements);
        dataStore.addData(hostData, numElements, elementSize, [](void *ptr) {
          delete static_cast<std::complex<double> *>(ptr);
        });
      }
    } else {
      auto hostData = state->get_tensor().data;
      dataStore.addData(hostData, numElements, elementSize, [](void *ptr) {});
    }
  }
  return dataStore;
}
} // namespace cudaq::runtime
