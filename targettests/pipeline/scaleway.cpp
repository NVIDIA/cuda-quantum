/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if nvq++ --list-targets | grep -qx scaleway; then \
// RUN:   nvq++ --enable-mlir --target scaleway --emulate %s -o %t && \
// RUN:   rm -f %t.log && \
// RUN:   CUDAQ_PIPELINE_LOG=%t.log %t && \
// RUN:   FileCheck %cudaq_src_dir/targettests/pipeline/log.checks --input-file=%t.log; \
// RUN: fi
// clang-format on

#include "trivial_kernel.h"
