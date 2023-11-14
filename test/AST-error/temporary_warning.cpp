/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -verify

#include <cudaq.h>

__qpu__ void t() {
  cudaq::qreg q(4);
  // expected-warning@+1{{If the intention is to use additional}}
  x(q[0], q[1], q[2]);
}
