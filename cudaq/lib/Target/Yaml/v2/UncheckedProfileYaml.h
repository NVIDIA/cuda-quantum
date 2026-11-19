/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Yaml/v2/ProfileYaml.h"
#include "cudaq/Target/Profile/UncheckedProfile.h"

/// YAML form of `UncheckedProfile`, which takes no options.
template <>
struct cudaq::config::v2::ProfileYaml<cudaq::config::UncheckedProfile> {
  struct Yaml {
    using Tag = rfl::Literal<UncheckedProfile::name>;
  };
  static UncheckedProfile to(const Yaml &) { return {}; }
  static Yaml from(const UncheckedProfile &) { return {}; }
};
