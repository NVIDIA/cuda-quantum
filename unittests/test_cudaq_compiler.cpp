/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq/nvqlink/compiler.h"
#include "cudaq/nvqlink/device.h"
#include "cudaq/nvqlink/nvqlink.h"

using namespace cudaq;

const std::string quake = R"#(
module attributes {cc.sizeof_string = 32 : i64, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.triple = "x86_64-unknown-linux-gnu", quake.mangled_name_map = {__nvqpp__mlirgen__function_test._Z4testii = "_Z4testii"}} {  
  func.func @__nvqpp__mlirgen__function_test._Z4testii(%arg0: i32, %arg1: i32) -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %c33_i32 = arith.constant 33 : i32
    %0 = cc.alloca i32
    cc.store %arg0, %0 : !cc.ptr<i32>
    %1 = cc.alloca i32
    cc.store %arg1, %1 : !cc.ptr<i32>
    %2 = cc.load %1 : !cc.ptr<i32>
    %3 = arith.addi %2, %c33_i32 : i32
    %4 = cc.load %0 : !cc.ptr<i32>
    %5 = arith.addi %3, %4 : i32
    return %5 : i32
  }
  func.func @_Z4testii(%arg0: i32, %arg1: i32) -> i32 attributes {no_this} {
    %0 = cc.undef i32
    return %0 : i32
  }
})#";

// Create the expected thunk args return structure
struct thunk_ret {
  void *ptr;
  std::size_t i;
};

TEST(NVQLinkQuakeSimCompilerTester, checkSimple) {

  // Test that we can compile Quake code!

  // Need a device to target
  nvqlink::nv_simulation_device device;

  // Get the Quake to Simulator compiler
  auto compiler = nvqlink::compiler::get("cudaq");

  // Compile, kernel name is "test", targeting one simulation device
  auto compiled = compiler->compile(quake, "function_test._Z4testii", 1);

  // Get the result, a binary program
  auto prog = compiled->get_programs()[0];

  // We know that the quake-sim compiler produces a binary
  // program consisting of a argsCreator and a thunk function
  // (enabling a uniform function signature interface)
  std::size_t (*argsCreator)(void **, void **);
  thunk_ret (*thunk)(void *, bool);

  // Extract those functions from the binary
  std::memcpy(&argsCreator, prog.binary.data(), sizeof(void *));
  std::memcpy(&thunk, prog.binary.data() + sizeof(void *), sizeof(void *));

  // Create the known thunk args struct
  struct thunk_struct {
    int i;
    int j;
    int k;
  };

  // Create some void** args for argsCreator
  int i = 2, j = 3;
  void *args[] = {&i, &j};
  thunk_struct *thunkArgs;

  // Map void** args to a void* thunk args
  argsCreator(args, reinterpret_cast<void **>(&thunkArgs));

  // argsCreator worked
  EXPECT_EQ(thunkArgs->i, 2);
  EXPECT_EQ(thunkArgs->j, 3);

  // Run the kernel
  thunk(thunkArgs, false);

  // Kernel returned the right value
  EXPECT_EQ(thunkArgs->k, 38);

  std::free(thunkArgs);
}
TEST(NVQLinkQuakeSimCompilerTester, checkCompileAndExecuteOnDevice) {

  // Need a device to target
  nvqlink::nv_simulation_device device;

  // Get the Quake to Simulator compiler
  auto compiler = nvqlink::compiler::get("cudaq");

  // Compile, kernel name is "test", targeting one simulation device
  auto compiled = compiler->compile(quake, "function_test._Z4testii", 1);

  // Get the result, a binary program
  auto prog = compiled->get_programs()[0];

  int i = 4, j = 5;
  auto result = nvqlink::malloc(sizeof(int));
  // want these by value
  device_ptr<int> iPtr(&i);
  device_ptr<int> jPtr(&j);

  device.upload_program(prog.binary);
  device.trigger(result, {iPtr, jPtr});

  EXPECT_EQ(*reinterpret_cast<int *>(result.handle), 42);

  nvqlink::free(result);
}
