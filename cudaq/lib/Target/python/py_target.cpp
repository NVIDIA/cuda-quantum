/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_compile_target.h"
#include "py_profiles.h"

NB_MODULE(_target, m) {
  cudaq::bindTargetProfiles(m);
  cudaq::bindCompileTarget(m);
}
