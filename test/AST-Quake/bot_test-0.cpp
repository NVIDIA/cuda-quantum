/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -std=c++20 %s | FileCheck %s

#include <cudaq.h>

struct Mate {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
  }
};

// CHECK: Mate
