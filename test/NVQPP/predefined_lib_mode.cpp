/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// clang-format off
// RUN: nvq++ -DTEST_DEF -DMY_VAR=\"CUDAQ\" %s -o out_test_predefined_lib_mode.x && ./out_test_predefined_lib_mode.x | FileCheck %s
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
