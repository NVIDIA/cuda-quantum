/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s -o /dev/null

#include <cudaq.h>
#include <set>

struct InvalidKernel1 {
  // expected-error@+1{{kernel argument type not supported}}
  void operator()(void *m) __qpu__ {
    cudaq::qvector reg(4);
    x(reg);
  }
};

struct InvalidKernel2 {
  // expected-error@+1{{kernel argument type not supported}}
  void operator()(int *m) __qpu__ {
    cudaq::qvector reg(4);
    x(reg);
  }
};

struct InvalidKernel2_1 {
  // expected-error@+1{{kernel argument type not supported}}
  void operator()(int &m) __qpu__ {
    cudaq::qvector reg(4);
    x(reg);
  }
};

struct InvalidKernel2_2 {
  // expected-error@+1{{kernel argument type not supported}}
  void operator()(const int &m) __qpu__ {
    cudaq::qvector reg(4);
    x(reg);
  }
};

struct InvalidKernel3_1 {
  // expected-error@+1{{kernel argument type not supported}}
  void operator()(const std::set<int> &m) __qpu__ {
    cudaq::qvector reg(4);
    x(reg);
  }
};

struct InvalidKernel4 {
  // expected-error@+1{{kernel result type not supported}}
  std::vector<int *> operator()() __qpu__ {
    cudaq::qvector reg(4);
    x(reg);
    return {};
  }
};
