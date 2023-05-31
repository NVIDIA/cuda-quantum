/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir  %s -o include.x && ./include.x | FileCheck %s

#include "include/include.h"
#include <iostream>

int main() {
  std::cout << MESSAGE << '\n';
  return RETURN_CODE;
}

// CHECK: Hello, World!
