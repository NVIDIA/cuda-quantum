/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <stdio.h>

extern "C" int add(int i, int j) {
  printf("Calling from the add function!!! add(%d,%d) = %d\n", i, j, i + j);
  return i + j;
}
