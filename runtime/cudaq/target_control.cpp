/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "target_control.h"

namespace cudaq::__internal__ {
static bool targetModificationKillSwitch = false;
void enableTargetModification() { targetModificationKillSwitch = true; }
void disableTargetModification() { targetModificationKillSwitch = false; }
bool canModifyTarget() { return targetModificationKillSwitch; }
} // namespace cudaq::__internal__