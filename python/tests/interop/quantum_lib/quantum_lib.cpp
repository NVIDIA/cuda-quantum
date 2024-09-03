/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "quantum_lib.h"

namespace cudaq {
__qpu__ void
entryPoint(const std::function<void(cudaq::qvector<> &)> &statePrep) {
  cudaq::qvector q(2);
  statePrep(q);
}
} // namespace cudaq