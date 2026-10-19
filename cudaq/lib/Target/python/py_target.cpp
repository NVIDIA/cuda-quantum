// Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "py_compile_target.h"
#include "py_profiles.h"

NB_MODULE(_target, m) {
  cudaq::bindTargetProfiles(m);
  cudaq::bindCompileTarget(m);
}
