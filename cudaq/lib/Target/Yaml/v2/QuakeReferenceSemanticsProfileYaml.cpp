/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v2/QuakeReferenceSemanticsProfileYaml.h"

using namespace cudaq::config;

QuakeReferenceSemanticsProfile
v2::ProfileYaml<QuakeReferenceSemanticsProfile>::to(const Yaml &yaml) {
  const auto &options = yaml.options.value();
  return {.basisGates = options.basisGates.value(),
          .measurementType = options.measurementType.value(),
          .resetAfterMeasure = options.resetAfterMeasure.value()};
}

v2::ProfileYaml<QuakeReferenceSemanticsProfile>::Yaml
v2::ProfileYaml<QuakeReferenceSemanticsProfile>::from(
    const QuakeReferenceSemanticsProfile &profile) {
  return {.options =
              Yaml::Options{.basisGates = profile.basisGates,
                            .measurementType = profile.measurementType,
                            .resetAfterMeasure = profile.resetAfterMeasure}};
}
