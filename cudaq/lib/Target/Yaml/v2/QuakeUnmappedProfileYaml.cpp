/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v2/QuakeUnmappedProfileYaml.h"

using namespace cudaq::config;

QuakeUnmappedProfile
v2::ProfileYaml<QuakeUnmappedProfile>::to(const Yaml &yaml) {
  const auto &options = yaml.options.value();
  return {.basisGates = options.basisGates.value(),
          .approximationTolerance = options.approximationTolerance.value(),
          .costModel = options.costModel,
          .semantics = options.semantics.value(),
          .unrollLoops = options.unrollLoops.value(),
          .maxUnrollIterations = options.maxUnrollIterations.value()};
}

v2::ProfileYaml<QuakeUnmappedProfile>::Yaml
v2::ProfileYaml<QuakeUnmappedProfile>::from(
    const QuakeUnmappedProfile &profile) {
  return {.options = Yaml::Options{
              .basisGates = profile.basisGates,
              .approximationTolerance = profile.approximationTolerance,
              .costModel = profile.costModel,
              .semantics = profile.semantics,
              .unrollLoops = profile.unrollLoops,
              .maxUnrollIterations = profile.maxUnrollIterations}};
}
