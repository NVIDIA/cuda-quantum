/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -Xcudaq -Wall -verify %s -o /dev/null

#include <cudaq.h>

void foo();

struct S {
   void operator()() __qpu__ {
      int unused = 42; // expected-warning{{unused variable}}
      foo();
   }
};
