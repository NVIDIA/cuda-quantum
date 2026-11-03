/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v2/QuakeMappedProfileYaml.h"

using namespace cudaq::config;

QuakeMappedProfile v2::ProfileYaml<QuakeMappedProfile>::to(const Yaml &yaml) {
  const auto &options = yaml.options.value();
  return {.basisGates = options.basisGates.value(), .device = options.device};
}

v2::ProfileYaml<QuakeMappedProfile>::Yaml
v2::ProfileYaml<QuakeMappedProfile>::from(const QuakeMappedProfile &profile) {
  return {.options = Yaml::Options{.basisGates = profile.basisGates,
                                   .device = profile.device}};
}
