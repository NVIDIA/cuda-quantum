/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is adapted from tweedledum library:
// https://github.com/boschmitt/tweedledum/blob/master/src/Utils/Visualization/string_utf8.cpp

#include "cudaq/algorithms/detectors.h"
#include <vector>

using namespace cudaq;

namespace cudaq::__internal__ {

std::vector<std::vector<std::int64_t>>
traceToDetectorMzIndices(const Trace &trace) {
  std::vector<std::vector<std::int64_t>> detectorMzIndices;
  int64_t measurement_counter = 0;
  for (const auto &instruction : trace) {
    if (instruction.name == "mx" || instruction.name == "my" ||
        instruction.name == "mz") {
      measurement_counter++;
    } else if (instruction.name == "detector") {
      std::vector<std::int64_t> detectorIndices;
      for (const auto &param : instruction.params) {
        if (param < 0) {
          if (measurement_counter + static_cast<std::int64_t>(param) < 0) {
            throw std::runtime_error("Detector measurement index is negative");
          }
          detectorIndices.push_back(measurement_counter +
                                    static_cast<std::int64_t>(param));
        } else {
          detectorIndices.push_back(-static_cast<std::int64_t>(param));
        }
      }
      detectorMzIndices.push_back(std::move(detectorIndices));
    }
  }
  return detectorMzIndices;
}
} // namespace cudaq::__internal__
