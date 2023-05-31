/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

/// Name of the attribute attached to entry point functions.
static constexpr char entryPointAttrName[] = "cudaq-entrypoint";

/// Name of the attribute attached to cudaq kernels.
static constexpr char kernelAttrName[] = "cudaq-kernel";

} // namespace cudaq
