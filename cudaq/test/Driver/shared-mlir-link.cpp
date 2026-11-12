/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The shared runtime must satisfy symbols before a competing static archive,
// including for consumers of the installed CMake target.
// clang-format off
// RUN: cmake -S %S/Inputs/shared_mlir_link -B %t.build -DCUDAQ_SOURCE_DIR=%cudaq_src_dir -DCMAKE_BUILD_TYPE=Release
// RUN: cmake --build %t.build --config Release
// RUN: ctest --test-dir %t.build -C Release --output-on-failure
// RUN: cmake --install %t.build --config Release --prefix %t.install
// RUN: cmake -S %S/Inputs/shared_mlir_link -B %t.consumer -DMLIR_TARGETS=%t.install/lib/cmake/SharedMLIR/SharedMLIRTargets.cmake -DCMAKE_BUILD_TYPE=Release
// RUN: cmake --build %t.consumer --config Release
// RUN: ctest --test-dir %t.consumer -C Release --output-on-failure
// clang-format on
