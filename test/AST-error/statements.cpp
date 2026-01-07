/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <iostream>

void foo();
void bar();

struct S2 {
  auto operator()() __qpu__ {
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{try statement is not yet supported}}
    try {
      foo();
    } catch (int) {
      bar();
    }
  }
};

struct S4 {
  auto operator()() __qpu__ {
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{goto statement}}
    goto label;
  label:
    foo();
  }
};

struct S5 {
  auto operator()(int sp) __qpu__ {
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{switch statement}}
    switch (sp) {
    case 1:
      foo();
    default:
      bar();
      break;
    }
  }
};

struct S6 {
  auto operator()() __qpu__ {
    // expected-error@*{{union types are not allowed in kernels}}
    // expected-error@+1{{statement not supported in qpu kernel}}
    std::cout << "Hello\n";

    // expected-error@+2{{cannot call variadic function from quantum kernel}}
    // expected-error@+1{{statement not supported in qpu kernel}}
    printf("Hello\n");
  }
};
