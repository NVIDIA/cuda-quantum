/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// This file provides some common cudaq execution manager function names for
/// use throughout our MLIR lowering infrastructure.

namespace cudaq::opt {

static constexpr const char CudaqEMAllocate[] = "__nvqpp__cudaq_em_allocate";
static constexpr const char CudaqEMAllocateVeq[] =
    "__nvqpp__cudaq_em_allocate_veq";
static constexpr const char CudaqEMApply[] = "__nvqpp__cudaq_em_apply";
static constexpr const char CudaqEMConcatSpan[] =
    "__nvqpp__cudaq_em_concatSpan";
static constexpr const char CudaqEMMeasure[] = "__nvqpp__cudaq_em_measure";
static constexpr const char CudaqEMReset[] = "__nvqpp__cudaq_em_reset";
static constexpr const char CudaqEMReturn[] = "__nvqpp__cudaq_em_return";
static constexpr const char CudaqEMWriteToSpan[] =
    "__nvqpp__cudaq_em_writeToSpan";

} // namespace cudaq::opt
