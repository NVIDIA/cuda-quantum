/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: not cudaq-quake %s 2>&1 | FileCheck %s

#include "cudaq/qis/qubit_qis.h"

// Kernel classes with data members: the members must be scalars, and the class
// may not have other member functions.

struct C {
  void operator()(cudaq::qubit &q) __qpu__ { h(q); }

  double supported;
};

struct Q {
  void operator()() __qpu__ {}

  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member may not have a quantum type
  cudaq::qubit q;
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member may not have a quantum type
  cudaq::qvector<> v;
};

struct R {
  void operator()() __qpu__ {}

  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member must be of an arithmetic or enumeration type
  int &ref;
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member must be of an arithmetic or enumeration type
  double *ptr;
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member must be of an arithmetic or enumeration type
  std::vector<int> vec;
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member must be of an arithmetic or enumeration type
  unsigned bits : 3;
};

struct M {
  M() = default;
  M(const M &) = default;
  M &operator=(const M &) = default;
  ~M() = default;
  void operator()() __qpu__ {}
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class with data members may not have other member functions
  int helper() { return i; }
  int i = 1;
};

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: class inheritance is not allowed for CUDA-Q kernel
struct D : public C {
  void operator()(cudaq::qubit &q) __qpu__ { h(q); }
};
// clang-format on
