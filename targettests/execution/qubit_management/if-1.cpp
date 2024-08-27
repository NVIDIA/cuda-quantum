/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// TODO: filecheck with statistics
// TODO: should work properly with regtomem fixes

// RUN: nvq++ --enable-mlir -fno-lower-to-cfg --opt-pass 'func.func(add-dealloc,combine-quantum-alloc,canonicalize,factor-quantum-alloc,memtoreg),canonicalize,cse,add-wireset,func.func(assign-wire-indices),dep-analysis,func.func(lower-to-cfg,regtomem),symbol-dce'  %s -o %t && %t

// Simple test, shouldn't affect anything
struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    if (true)
      x(q);
    else
      y(q);

    bool b = mz(q);

    return b;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
