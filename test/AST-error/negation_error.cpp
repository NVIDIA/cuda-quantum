/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct NegationOperatorTest {
  void operator()() __qpu__ {
    cudaq::qreg qr(3);
    x<cudaq::ctrl>(qr[0], qr[1], !qr[2]); // expected-error{{target qubit cannot be negated}}
    rz(2.0, !qr[0]); // expected-error{{target qubit cannot be negated}}
  }
};
