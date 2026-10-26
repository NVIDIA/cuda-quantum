/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq::runtime {

static constexpr const char mangledNameMap[] = "quake.mangled_name_map";

static constexpr const char sizeofStringAttrName[] = "cc.sizeof_string";

static constexpr const char enableCudaqRun[] = "quake.cudaq_run";

static constexpr const char pythonUniqueAttrName[] = "quake.python_uniqued";

static constexpr const char disableQuantumOpts[] = "quake.noOptimization";

static constexpr const char operandSegmentSizes[] = "operandSegmentSizes";

} // namespace cudaq::runtime
