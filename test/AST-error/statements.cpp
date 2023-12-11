/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
    try { // expected-error   {{try statement is not yet supported}}
          // expected-error@-1{{statement not supported in qpu kernel}}
      foo();
    } catch (int) {
      bar();
    }
  }
};

struct S4 {
  auto operator()() __qpu__ {
    goto label; // expected-error   {{goto statement}}
                // expected-error@-1{{statement not supported in qpu kernel}}
  label:
    foo();
  }
};

struct S5 {
  auto operator()(int sp) __qpu__ {
    switch (sp) { // expected-error   {{switch statement}}
                  // expected-error@-1{{statement not supported in qpu kernel}}
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
    // expected-error@+1{{statement not supported in qpu kernel}}
    std::cout << "Hello\n";

    // expected-error@+2{{cannot call variadic function from quantum kernel}}
    // expected-error@+1{{statement not supported in qpu kernel}}
    printf("Hello\n");
  }
};
