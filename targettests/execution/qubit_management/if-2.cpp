/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ --enable-mlir --opt-pass 'func.func(add-dealloc,combine-quantum-alloc,canonicalize,factor-quantum-alloc,memtoreg),canonicalize,cse,add-wireset,func.func(assign-wire-indices),dep-analysis,func.func(regtomem),symbol-dce'  %s -o %t && %t

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    bool res;

    h(q);
    bool b = mz(q);
    
    // Should be able to lift x(p/r)
    if (b) {
      cudaq::qubit p;
      x(p);
      y(p);
      res = mz(p);
    } else {
      cudaq::qubit r;
      x(r);
      z(r);
      res = mz(r);
    }

    return res;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
