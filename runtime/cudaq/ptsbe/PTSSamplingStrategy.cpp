/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSSamplingStrategy.h"
#include "common/NoiseModel.h"
#include <numeric>

namespace cudaq::ptsbe {

bool NoisePoint::isUnitaryMixture(double tolerance) const {
  if (kraus_operators.empty()) {
    double sum = 0.0;
    for (auto p : probabilities)
      sum += p;
    return std::abs(sum - 1.0) < tolerance;
  }
  
  if (probabilities.size() != kraus_operators.size())
    return false;
  
  auto validated_result = cudaq::computeUnitaryMixture(kraus_operators, tolerance);
  
  if (!validated_result.has_value())
    return false;
  
  const auto& validated_probs = validated_result->first;
  
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    if (std::abs(probabilities[i] - validated_probs[i]) > tolerance)
      return false;
  }
  
  return true;
}

} // namespace cudaq::ptsbe
