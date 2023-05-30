/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o introspection.x && ./introspection.x | grep lookhere | FileCheck %s

#include <iostream>
#include <cudaq.h>

struct applyH {
  template <typename A> void operator()(cudaq::qubit &q, A a) __qpu__ { h(q); }
};

struct dummy {
  void operator()() __qpu__ {
    cudaq::qubit q;
    applyH{}(q, 42);
  }
};

template <typename T> __qpu__ void funky(T i) {}

// Without weak, this function has multiple definitions and the linker raises an
// error.
__attribute__((weak)) __qpu__ void funny(int i) { funky(i); }

template <typename T> struct Qernel {
  template <typename A> void operator()(cudaq::qubit &q, T t, A a) __qpu__ {
    h(q);
  }
};

struct QernelEntry {
  void operator()() __qpu__ {
    cudaq::qubit q;
    Qernel<double>{}(q, 3.14, 42);
  }
};

int main() {
  auto quake = cudaq::get_quake<int>(applyH{});
  std::cout << "lookhere 1: " << quake << '\n';
  quake = cudaq::get_quake(std::string("funny"));
  std::cout << "lookhere 2: " << quake << '\n';
  quake = cudaq::get_quake<int>("funky");
  std::cout << "lookhere 3: " << quake << '\n';
  quake = cudaq::get_quake<int>(Qernel<double>{});
  std::cout << "lookhere 4: " << quake << '\n';
  return 0;
}

// CHECK: 1: {{.*}} @__nvqpp__mlirgen__instance_applyHi
// CHECK: 2: {{.*}} @__nvqpp__mlirgen__function_funny
// CHECK: 3: {{.*}} @__nvqpp__mlirgen__instance_function_funkyi
// CHECK: 4: {{.*}} @__nvqpp__mlirgen__instance_QernelIdEi
