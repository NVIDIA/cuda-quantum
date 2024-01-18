/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: if [ $(echo %cpp_std | cut -c4- ) -ge 20 ]; then \
// RUN:   nvq++ --enable-mlir %s -o %t && %t | FileCheck %s; \
// RUN: fi

#include "include/include.h"
#include <iostream>

int main() {
  std::cout << MESSAGE << '\n';
  return RETURN_CODE;
}

// CHECK: Hello, World!
