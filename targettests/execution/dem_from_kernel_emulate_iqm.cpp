/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target iqm --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_20.txt %t | FileCheck %s
// RUN: nvq++ --target iqm --emulate --mapping-file %iqm_tests_dir/Crystal_20.txt %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm --emulate %s -o %t && IQM_SERVER_URL=fake-fake-fake CUDAQ_LOG_LEVEL=warn %t 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=INVALIDURL %s < %t.err

#include "dem_from_kernel_emulate.cpp"

// CHECK: SINGLE detectors=1 observables=0
// CHECK: THREE_MZ detectors=1 observables=1
// CHECK: CORRELATED_XX_RAW hyperedge=1 caret=0
// CHECK: CORRELATED_XX_DECOMPOSED hyperedge=0 caret=1
// CHECK: SAMPLE_QEC_KERNEL most_probable=0

// INVALIDURL: Leaving %QPU_ARCH% unresolved: Unable to get quantum architecture from
// INVALIDURL-SAME: fake-fake-fake
// INVALIDURL: Path %QPU_ARCH% does not exist
// clang-format on
