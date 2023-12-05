/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir -shared -fpic %s -o %t && file %t | FileCheck %s

#include <cudaq.h>

struct Qernel {
  void operator()() __qpu__ {}
};

int plain_old_function() { return 0; }

// CHECK: shared
