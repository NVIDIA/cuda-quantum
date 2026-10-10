/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EstimateResult.h"
#include "common/SampleResult.h"
#include "nlohmann/json.hpp"

int main() {
  cudaq::ExecutionResult samples(cudaq::CountsDictionary{{"0", 10}});
  auto serialized = samples.serialize();
  cudaq::ExecutionResult restored;
  restored.deserialize(serialized);

  cudaq::estimate_result estimate;
  estimate.get_annotations().get()["shots"] = 10;
  cudaq::estimate_result copy(estimate);
  return !(restored == samples && copy.get_annotations().get()["shots"] == 10);
}
