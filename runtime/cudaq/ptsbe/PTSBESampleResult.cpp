/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESampleResult.h"

namespace cudaq::ptsbe {

sample_result::sample_result(cudaq::sample_result &&base)
    : cudaq::sample_result(std::move(base)) {}

} // namespace cudaq::ptsbe
