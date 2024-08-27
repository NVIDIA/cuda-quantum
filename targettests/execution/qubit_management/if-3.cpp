/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ --enable-mlir -fno-lower-to-cfg --opt-pass 'func.func(add-dealloc,combine-quantum-alloc,canonicalize,factor-quantum-alloc,memtoreg),canonicalize,cse,add-wireset,func.func(assign-wire-indices),dep-analysis,func.func(lower-to-cfg,regtomem),symbol-dce'  %s -o %t && %t

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    h(q);
    bool b = mz(q);
    // Should be able to lift x(q) after
    if (b) {
      y(q);
      x(q);
    } else {
      h(q);
      x(q);
    }

    bool res = mz(q);
    return res;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
