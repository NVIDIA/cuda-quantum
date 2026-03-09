/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// clang-format off
// RUN: nvq++ -DTEST_DEF -DMY_VAR=\"CUDAQ\" %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir -DTEST_DEF -DMY_VAR=\"CUDAQ\" %s -o %t
// clang-format on

#include <iostream>

int main() {
#if defined(TEST_DEF)
  std::cout << "PASS\n";
#else
  std::cout << "FAIL\n";
#endif
  // CHECK: PASS

#if defined(MY_VAR)
  std::cout << MY_VAR << "\n";
#else
  std::cout << "FAIL\n";
#endif
  // CHECK: CUDAQ
  return 0;
}
