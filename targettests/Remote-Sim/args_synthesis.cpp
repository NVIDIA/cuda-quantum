/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// clang-format off
// RUN: nvq++ --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// RUN: nvq++ --enable-mlir --no-aggressive-early-inline --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

// This is a comprehensive set of tests for kernel argument synthesis for remote
// platforms. Note: we use the remote-mqpu platform in MLIR mode as a mock
// environment for NVQC.

// The test cases are defined in these headers.
// Comment out include lines to exclude tests for debugging.
#include "args_synthesis_test_cases/1_single_arg_tests.h"
#include "args_synthesis_test_cases/2_two_pod_args_tests.h"
#include "args_synthesis_test_cases/3_two_args_pod_container.h"
#include "args_synthesis_test_cases/4_two_container_args.h"
#include "args_synthesis_test_cases/5_mixed_args_tests.h"

int main() {
  // Run all tests
  for (auto &functor : ALL_TEST_FUNCTORS)
    functor();
  return 0;
}
