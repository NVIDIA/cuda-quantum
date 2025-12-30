/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify -D CUDAQ_REMOTE_SIM=1 %s

#include <cudaq.h>

__qpu__ void k() {
  // expected-error@+2 {{variable has invalid storage class}}
  // expected-error@+1 {{statement not supported}}
  static int i = 0;
}

__qpu__ void l() {
  // expected-error@+2 {{variable has invalid storage class}}
  // expected-error@+1 {{statement not supported}}
  extern int i;
}
