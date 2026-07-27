/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// nvq++ driver test (for its whole pipeline) for a kernel that returns a vector
// of bools. The test checks that the nvq++ optimization pipeline does not
// introduce any llvm.memcpy or malloc calls when returning a vector of bools
// from a kernel.
//
// Compile in an isolated directory with -save-temps and check the generated
// .qke file, which is the Quake IR after the nvq++ optimization pipeline.
//
// RUN: rm -rf %t && mkdir %t
// RUN: cd %t && nvq++ -save-temps %s -o app
// RUN: FileCheck %s --input-file=%t/run_vector_return.dcl.qke

#include <cudaq.h>

struct run_vector_return {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(1);
    std::vector<bool> result(1);
    x(q[0]);
    result[0] = mz(q[0]);
    return result;
  }
};

int main() {
  auto results = cudaq::run(1, run_vector_return{});
  return results[0][0] ? 0 : 1;
}

// `run-semantics-hackery` removes the vector copy constructor used to return
// the result. Without the pass, aggressive inlining lowers that copy to malloc
// and llvm.memcpy in the generated `.run` function. Check that neither
// operation is present before the returned value is logged.
//
// CHECK-LABEL: func.func @__nvqpp__mlirgen__run_vector_return.run()
// CHECK-NOT:   call @malloc
// CHECK-NOT:   call @llvm.memcpy
// CHECK:       cc.log_output
// CHECK:       return
