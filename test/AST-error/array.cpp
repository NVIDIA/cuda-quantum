/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct T {
   void operator()(int N) __qpu__ {
      cudaq::qreg Q(N);
      x(Q);
   }
};

struct S {
   void operator()() __qpu__ {
      int arr[3];
      T{}(arr[0]); // expected-error{{arrays in kernel}}
      T{}(arr[1]);
      T{}(arr[2]);
   }
};
