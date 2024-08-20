/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ --enable-mlir --opt-pass 'func.func(add-dealloc,combine-quantum-alloc,canonicalize,factor-quantum-alloc,memtoreg),canonicalize,cse,add-wireset,func.func(assign-wire-indices),dep-analysis,func.func(regtomem),symbol-dce'  %s -o %t && %t
// XFAIL: *

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit p;

    if (true) {
      rx(1., p);
      z(p);
      rx(1., p);
    } else {
      rx(1., p);
      y(p);
    }
    auto res = mz(p);
    return res;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
