/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

void foo();
void bar();

struct S2 {
  auto operator()() __qpu__ {
    try { // expected-error{{try statement is not yet supported}}
      foo();
    } catch (int) {
      bar();
    }
  }
};

struct S4 {
  auto operator()() __qpu__ {
    goto label; // expected-error{{goto statement}}
  label:
    foo();
  }
};

struct S5 {
  auto operator()(int sp) __qpu__ {
    switch (sp) { // expected-error{{switch statement}}
    case 1:
      foo();
    default:
      bar();
      break;
    }
  }
};
