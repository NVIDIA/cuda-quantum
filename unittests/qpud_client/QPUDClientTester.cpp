/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/utils/registry.h"

#include <cudaq.h>

#include "qpud_client.h"

TEST(QPUDClientTester, checkSample) {

  const std::string_view quakeCode =
      R"#(module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__ghz = "_ZN3ghzclEi"}} {
  func.func @__nvqpp__mlirgen__ghz(%arg0: i32) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloca() : memref<i32>
    memref.store %arg0, %0[] : memref<i32>
    %1 = memref.load %0[] : memref<i32>
    %2 = arith.extsi %1 : i32 to i64
    %3 = quake.alloca(%2 : i64) : !quake.qvec<?>
    %4 = quake.qextract %3[%c0_i64] : !quake.qvec<?>[i64]  -> !quake.qref
    quake.h (%4)
    cc.scope {
      %9 = memref.alloca() : memref<i32>
      memref.store %c0_i32, %9[] : memref<i32>
      cc.loop while {
        %10 = memref.load %9[] : memref<i32>
        %11 = memref.load %0[] : memref<i32>
        %12 = arith.subi %11, %c1_i32 : i32
        %13 = arith.cmpi slt, %10, %12 : i32
        cc.condition %13
      } do {
        cc.scope {
          %10 = memref.load %9[] : memref<i32>
          %11 = arith.extsi %10 : i32 to i64
          %12 = quake.qextract %3[%11] : !quake.qvec<?>[i64] -> !quake.qref
          %13 = memref.load %9[] : memref<i32>
          %14 = arith.addi %13, %c1_i32 : i32
          %15 = arith.extsi %14 : i32 to i64
          %16 = quake.qextract %3[%15] : !quake.qvec<?>[i64] -> !quake.qref
          quake.x [%12 : !quake.qref] (%16)
        }
        cc.continue
      } step {
        %10 = memref.load %9[] : memref<i32>
        %11 = arith.addi %10, %c1_i32 : i32
        memref.store %11, %9[] : memref<i32>
      }
    }
    %5 = quake.qvec_size %3 : (!quake.qvec<?>) -> i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = llvm.alloca %5 x i1 : (i64) -> !llvm.ptr<i1>
    affine.for %arg1 = 0 to %6 {
      %9 = quake.qextract %3[%arg1] : !quake.qvec<?>[index] -> !quake.qref
      %10 = quake.mz(%9 : !quake.qref) : i1
      %11 = arith.index_cast %arg1 : index to i64
      %12 = llvm.getelementptr %7[%11] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %10, %12 : !llvm.ptr<i1>
    }
    %8 = cc.stdvec_init %7, %5 : (!llvm.ptr<i1>, i64) -> !cc.stdvec<i1>
    return
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !llvm.struct<(ptr<i8>, i64)> {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<i8>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i8>, i64)> 
    %3 = llvm.insertvalue %c0_i64, %2[1] : !llvm.struct<(ptr<i8>, i64)> 
    return %3 : !llvm.struct<(ptr<i8>, i64)>
  }
  func.func @ghz.thunk(%arg0: !llvm.ptr<i8>, %arg1: i1) -> !llvm.struct<(ptr<i8>, i64)> {
    %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(i32)>>
    %1 = llvm.load %0 : !llvm.ptr<struct<(i32)>>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<struct<(i32)>>
    %4 = llvm.getelementptr %3[1] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<struct<(i32)>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<struct<(i32)>> to i64
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.extractvalue %1[0] : !llvm.struct<(i32)> 
    call @__nvqpp__mlirgen__ghz(%7) : (i32) -> ()
    %nil = call @__nvqpp_zeroDynamicResult() : () -> !llvm.struct<(ptr<i8>, i64)>
    return %nil : !llvm.struct<(ptr<i8>, i64)>
  }
})#";

  std::size_t shots = 500;
  cudaq::registry::deviceCodeHolderAdd("ghz", quakeCode.data());

  // Here is the main qpud_client sampling workflow

  // Create the client
  cudaq::qpud_client client;

  // Create a struct defining the runtime args for the kernel
  struct KernelArgs {
    int N = 5;
  } args;

  // Map those args to a void pointer and its associated size
  auto [rawArgs, size, resultOff] = client.process_args(args);

  // Invoke the sampling workflow, get the MeasureCounts
  auto counts = client.sample("ghz", shots, rawArgs, size);

  // Test the results.
  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "00000" || bits == "11111");
  }
  EXPECT_EQ(counter, shots);

  counts.dump();

  // Try it again with the simpler API
  counts = client.sample("ghz", shots, args);
  counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "00000" || bits == "11111");
  }
  EXPECT_EQ(counter, shots);
}

TEST(QPUDClientTester, checkObserve) {

  const std::string_view quakeCode =
      R"#(module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__ansatz = "_ZN6ansatzclEd"}} {
  func.func @__nvqpp__mlirgen__ansatz(%arg0: f64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = memref.alloca() : memref<f64>
    memref.store %arg0, %0[] : memref<f64>
    %1 = quake.alloca : !quake.qvec<2>
    %2 = quake.qextract %1[%c0_i64] : !quake.qvec<2>[i64] -> !quake.qref
    quake.x (%2)
    %3 = memref.load %0[] : memref<f64>
    %4 = quake.qextract %1[%c1_i64] : !quake.qvec<2>[i64] -> !quake.qref
    quake.ry |%3 : f64|(%4)
    %5 = quake.qextract %1[%c1_i64] : !quake.qvec<2>[i64] -> !quake.qref
    %6 = quake.qextract %1[%c0_i64] : !quake.qvec<2>[i64] -> !quake.qref
    quake.x [%5 : !quake.qref] (%6)
    return
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !llvm.struct<(ptr<i8>, i64)> {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<i8>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i8>, i64)> 
    %3 = llvm.insertvalue %c0_i64, %2[1] : !llvm.struct<(ptr<i8>, i64)> 
    return %3 : !llvm.struct<(ptr<i8>, i64)>
  }
  func.func @ansatz.thunk(%arg0: !llvm.ptr<i8>, %arg1: i1) -> !llvm.struct<(ptr<i8>, i64)> {
    %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(f64)>>
    %1 = llvm.load %0 : !llvm.ptr<struct<(f64)>>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<struct<(f64)>>
    %4 = llvm.getelementptr %3[1] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<struct<(f64)>> to i64
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.extractvalue %1[0] : !llvm.struct<(f64)> 
    call @__nvqpp__mlirgen__ansatz(%7) : (f64) -> ()
    %nil = call @__nvqpp_zeroDynamicResult() : () -> !llvm.struct<(ptr<i8>, i64)>
    return %nil : !llvm.struct<(ptr<i8>, i64)>
  }
})#";

  cudaq::registry::deviceCodeHolderAdd("ansatz", quakeCode.data());

  // Here is the main qpud_client sampling workflow

  // Create the client
  cudaq::qpud_client client;

  // Create a struct defining the runtime args for the kernel
  struct KernelArgs {
    double theta = 0.59;
  } args;

  // Map those args to a void pointer and its associated size
  auto [rawArgs, size, resultOff] = client.process_args(args);

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  double expVal = client.observe("ansatz", h, rawArgs, size);

  EXPECT_NEAR(expVal, -1.74, 1e-2);

  // Try it again with the simpler API
  expVal = client.observe("ansatz", h, args);
  EXPECT_NEAR(expVal, -1.74, 1e-2);
}

TEST(QPUDClientTester, checkExecute) {

  const std::string_view quakeCode =
      R"#(module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__super = "_ZN5superclEd"}} {
  func.func @__nvqpp__mlirgen__super(%arg0: f64) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %alloca = memref.alloca() : memref<f64>
    memref.store %arg0, %alloca[] : memref<f64>
    %0 = quake.alloca : !quake.qref
    %1 = memref.load %alloca[] : memref<f64>
    quake.rx |%1 : f64|(%0)
    %2 = memref.load %alloca[] : memref<f64>
    %cst = arith.constant 2.000000e+00 : f64
    %3 = arith.divf %2, %cst : f64
    quake.ry |%3 : f64|(%0)
    %4 = quake.mz(%0 : !quake.qref) : i1
    return %4 : i1
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !llvm.struct<(ptr<i8>, i64)> {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<i8>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i8>, i64)> 
    %3 = llvm.insertvalue %c0_i64, %2[1] : !llvm.struct<(ptr<i8>, i64)> 
    return %3 : !llvm.struct<(ptr<i8>, i64)>
  }
  func.func @super.thunk(%arg0: !llvm.ptr<i8>, %arg1: i1) -> !llvm.struct<(ptr<i8>, i64)> {
    %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(f64, i1)>>
    %1 = llvm.load %0 : !llvm.ptr<struct<(f64, i1)>>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<struct<(f64, i1)>>
    %4 = llvm.getelementptr %3[1] : (!llvm.ptr<struct<(f64, i1)>>) -> !llvm.ptr<struct<(f64, i1)>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<struct<(f64, i1)>> to i64
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.extractvalue %1[0] : !llvm.struct<(f64, i1)> 
    %8 = call @__nvqpp__mlirgen__super(%7) : (f64) -> i1
    %9 = llvm.getelementptr %0[0, 1] : (!llvm.ptr<struct<(f64, i1)>>) -> !llvm.ptr<i1>
    llvm.store %8, %9 : !llvm.ptr<i1>
    %10 = call @__nvqpp_zeroDynamicResult() : () -> !llvm.struct<(ptr<i8>, i64)>
    return %10 : !llvm.struct<(ptr<i8>, i64)>
  }
})#";

  cudaq::registry::deviceCodeHolderAdd("super", quakeCode.data());

  // Here is the main qpud_client sampling workflow

  // Create the client
  cudaq::qpud_client client;

  // Create a struct defining the runtime args for the kernel
  struct KernelArgs {
    double theta = M_PI;
    bool retVal;
  } args;

  // Map those args to a void pointer and its associated size
  auto [rawArgs, size, resultOff] = client.process_args(args);

  for (std::size_t i = 0; i < 10; i++) {
    client.execute("super", rawArgs, size, resultOff);
    auto retVal = args.retVal;
    EXPECT_TRUE(retVal == 0 || retVal == 1);
  }
}

std::size_t ghzArgsCreator(void **packedArgs, void **argMem) {
  struct KernelArgs {
    int N;
  };

  // This is freed by sample_detach...
  KernelArgs *args = new KernelArgs();
  args->N = *reinterpret_cast<int *>(packedArgs[0]);
  *argMem = args;
  return sizeof(KernelArgs);
}

TEST(QPUDClientTester, checkSampleDetached) {

  const std::string_view quakeCode =
      R"#(module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__ghz = "_ZN3ghzclEi"}} {
  func.func @__nvqpp__mlirgen__ghz(%arg0: i32) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloca() : memref<i32>
    memref.store %arg0, %0[] : memref<i32>
    %1 = memref.load %0[] : memref<i32>
    %2 = arith.extsi %1 : i32 to i64
    %3 = quake.alloca(%2 : i64) : !quake.qvec<?>
    %4 = quake.qextract %3[%c0_i64] : !quake.qvec<?>[i64] -> !quake.qref
    quake.h (%4)
    cc.scope {
      %9 = memref.alloca() : memref<i32>
      memref.store %c0_i32, %9[] : memref<i32>
      cc.loop while {
        %10 = memref.load %9[] : memref<i32>
        %11 = memref.load %0[] : memref<i32>
        %12 = arith.subi %11, %c1_i32 : i32
        %13 = arith.cmpi slt, %10, %12 : i32
        cc.condition %13
      } do {
        cc.scope {
          %10 = memref.load %9[] : memref<i32>
          %11 = arith.extsi %10 : i32 to i64
          %12 = quake.qextract %3[%11] : !quake.qvec<?>[i64] -> !quake.qref
          %13 = memref.load %9[] : memref<i32>
          %14 = arith.addi %13, %c1_i32 : i32
          %15 = arith.extsi %14 : i32 to i64
          %16 = quake.qextract %3[%15] : !quake.qvec<?>[i64] -> !quake.qref
          quake.x [%12 : !quake.qref] (%16)
        }
        cc.continue
      } step {
        %10 = memref.load %9[] : memref<i32>
        %11 = arith.addi %10, %c1_i32 : i32
        memref.store %11, %9[] : memref<i32>
      }
    }
    %5 = quake.qvec_size %3 : (!quake.qvec<?>) -> i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = llvm.alloca %5 x i1 : (i64) -> !llvm.ptr<i1>
    affine.for %arg1 = 0 to %6 {
      %9 = quake.qextract %3[%arg1] : !quake.qvec<?>[index] -> !quake.qref
      %10 = quake.mz(%9 : !quake.qref) : i1
      %11 = arith.index_cast %arg1 : index to i64
      %12 = llvm.getelementptr %7[%11] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %10, %12 : !llvm.ptr<i1>
    }
    %8 = cc.stdvec_init %7, %5 : (!llvm.ptr<i1>, i64) -> !cc.stdvec<i1>
    return
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !llvm.struct<(ptr<i8>, i64)> {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<i8>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i8>, i64)> 
    %3 = llvm.insertvalue %c0_i64, %2[1] : !llvm.struct<(ptr<i8>, i64)> 
    return %3 : !llvm.struct<(ptr<i8>, i64)>
  }
  func.func @ghz.thunk(%arg0: !llvm.ptr<i8>, %arg1: i1) -> !llvm.struct<(ptr<i8>, i64)> {
    %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(i32)>>
    %1 = llvm.load %0 : !llvm.ptr<struct<(i32)>>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<struct<(i32)>>
    %4 = llvm.getelementptr %3[1] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<struct<(i32)>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<struct<(i32)>> to i64
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.extractvalue %1[0] : !llvm.struct<(i32)> 
    call @__nvqpp__mlirgen__ghz(%7) : (i32) -> ()
    %nil = call @__nvqpp_zeroDynamicResult() : () -> !llvm.struct<(ptr<i8>, i64)>
    return %nil : !llvm.struct<(ptr<i8>, i64)>
  }
})#";

  std::size_t shots = 500;
  cudaq::registry::deviceCodeHolderAdd("ghz", quakeCode.data());
  std::size_t (*ptr)(void **, void **);
  ptr = ghzArgsCreator;
  cudaq::registry::cudaqRegisterArgsCreator("ghz",
                                            reinterpret_cast<char *>(ptr));

  // Here is the main qpud_client sampling workflow

  // Create the client
  auto job = cudaq::sample_detach("ghz", shots, 5);
  std::cout << job[0].id << "\n";

  auto counts = cudaq::sample(job);
  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "00000" || bits == "11111");
  }
  EXPECT_EQ(counter, shots);

  counts.dump();
}

std::size_t ansatzArgsCreator(void **packedArgs, void **argMem) {
  struct KernelArgs {
    double theta;
  };

  KernelArgs *args = new KernelArgs();
  args->theta = *reinterpret_cast<double *>(packedArgs[0]);
  *argMem = args;

  return sizeof(KernelArgs);
}

TEST(QPUDClientTester, checkObserveDetached) {

  const std::string_view quakeCode =
      R"#(module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__ansatz = "_ZN6ansatzclEd"}} {
  func.func @__nvqpp__mlirgen__ansatz(%arg0: f64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = memref.alloca() : memref<f64>
    memref.store %arg0, %0[] : memref<f64>
    %1 = quake.alloca : !quake.qvec<2>
    %2 = quake.qextract %1[%c0_i64] : !quake.qvec<2>[i64] -> !quake.qref
    quake.x (%2)
    %3 = memref.load %0[] : memref<f64>
    %4 = quake.qextract %1[%c1_i64] : !quake.qvec<2>[i64] -> !quake.qref
    quake.ry |%3 : f64|(%4)
    %5 = quake.qextract %1[%c1_i64] : !quake.qvec<2>[i64] -> !quake.qref
    %6 = quake.qextract %1[%c0_i64] : !quake.qvec<2>[i64] -> !quake.qref
    quake.x [%5 : !quake.qref] (%6)
    return
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !llvm.struct<(ptr<i8>, i64)> {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<i8>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i8>, i64)> 
    %3 = llvm.insertvalue %c0_i64, %2[1] : !llvm.struct<(ptr<i8>, i64)> 
    return %3 : !llvm.struct<(ptr<i8>, i64)>
  }
  func.func @ansatz.thunk(%arg0: !llvm.ptr<i8>, %arg1: i1) -> !llvm.struct<(ptr<i8>, i64)> {
    %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(f64)>>
    %1 = llvm.load %0 : !llvm.ptr<struct<(f64)>>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<struct<(f64)>>
    %4 = llvm.getelementptr %3[1] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<struct<(f64)>> to i64
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.extractvalue %1[0] : !llvm.struct<(f64)> 
    call @__nvqpp__mlirgen__ansatz(%7) : (f64) -> ()
    %nil = call @__nvqpp_zeroDynamicResult() : () -> !llvm.struct<(ptr<i8>, i64)>
    return %nil : !llvm.struct<(ptr<i8>, i64)>
  }
})#";

  cudaq::registry::deviceCodeHolderAdd("ansatz", quakeCode.data());
  std::size_t (*ptr)(void **, void **);
  ptr = ansatzArgsCreator;
  cudaq::registry::cudaqRegisterArgsCreator("ansatz",
                                            reinterpret_cast<char *>(ptr));

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  {
    auto jobs = cudaq::observe_detach("ansatz", h, 0.59);
    for (auto &j : jobs) {
      printf("%s %s\n", j.id.data(), j.name.data());
    }
    double res = cudaq::observe(h, jobs);

    EXPECT_NEAR(res, -1.74, 1e-2);
  }

  // Observe with shots
  {
    std::size_t shots = 10000;
    auto jobs = cudaq::observe_detach("ansatz", shots, h, 0.59);
    for (auto &j : jobs) {
      printf("%s %s\n", j.id.data(), j.name.data());
    }
    auto res = cudaq::observe(h, jobs);

    EXPECT_NEAR(res.exp_val_z(), -1.74, 1e-1);
    res.dump();
    auto x0x1Counts = res.counts(x(0) * x(1));
    x0x1Counts.dump();
    EXPECT_TRUE(x0x1Counts.size() == 4);

    auto z1Counts = res.counts(z(1));
    z1Counts.dump();

    EXPECT_EQ(2, z1Counts.size());
  }
}

std::size_t ansatzArgsCreatorVector(void **packedArgs, void **argMem) {
  struct KernelArgs {
    std::size_t size;
    double data;
  };

  // This is effectively what the automated code is doing.
  auto vector = *reinterpret_cast<std::vector<double> *>(packedArgs[0]);
  KernelArgs *args = new KernelArgs();
  args->size = sizeof(double);
  args->data = vector[0];
  *argMem = args;

  return sizeof(std::size_t) + sizeof(double);
}

TEST(QPUDClientTester, checkObserveDetachedWithVector) {

  const std::string_view quakeCode =
      R"#(module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__ansatzVec = "_ZN6ansatzVecclESt6vectorIdSaIdEE"}} {
  func.func @__nvqpp__mlirgen__ansatzVec(%arg0: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"} {
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.extsi %c2_i32 : i32 to i64
    %1 = quake.alloca(%0 : i64) : !quake.qvec<?>
    %c0_i32 = arith.constant 0 : i32
    %2 = arith.extsi %c0_i32 : i32 to i64
    %3 = quake.qextract %1[%2] : !quake.qvec<?>[i64] -> !quake.qref
    quake.x (%3)
    %c0_i32_0 = arith.constant 0 : i32
    %4 = arith.extsi %c0_i32_0 : i32 to i64
    %5 = cc.stdvec_data %arg0 : (!cc.stdvec<f64>) -> !llvm.ptr<f64>
    %6 = llvm.getelementptr %5[%4] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %7 = llvm.load %6 : !llvm.ptr<f64>
    %c1_i32 = arith.constant 1 : i32
    %8 = arith.extsi %c1_i32 : i32 to i64
    %9 = quake.qextract %1[%8] : !quake.qvec<?>[i64] -> !quake.qref
    quake.ry |%7 : f64|(%9)
    %c1_i32_1 = arith.constant 1 : i32
    %10 = arith.extsi %c1_i32_1 : i32 to i64
    %11 = quake.qextract %1[%10] : !quake.qvec<?>[i64] -> !quake.qref
    %c0_i32_2 = arith.constant 0 : i32
    %12 = arith.extsi %c0_i32_2 : i32 to i64
    %13 = quake.qextract %1[%12] : !quake.qvec<?>[i64] -> !quake.qref
    quake.x [%11 : !quake.qref] (%13)
    return
  }
  func.func private @__nvqpp_zeroDynamicResult() -> !llvm.struct<(ptr<i8>, i64)> {
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.inttoptr %c0_i64 : i64 to !llvm.ptr<i8>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i8>, i64)> 
    %3 = llvm.insertvalue %c0_i64, %2[1] : !llvm.struct<(ptr<i8>, i64)> 
    return %3 : !llvm.struct<(ptr<i8>, i64)>
  }
  func.func @ansatzVec.thunk(%arg0: !llvm.ptr<i8>, %arg1: i1) -> !llvm.struct<(ptr<i8>, i64)> {
    %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(i64)>>
    %1 = llvm.load %0 : !llvm.ptr<struct<(i64)>>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<struct<(i64)>>
    %4 = llvm.getelementptr %3[1] : (!llvm.ptr<struct<(i64)>>) -> !llvm.ptr<struct<(i64)>>
    %5 = llvm.ptrtoint %4 : !llvm.ptr<struct<(i64)>> to i64
    %6 = llvm.getelementptr %arg0[%5] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.extractvalue %1[0] : !llvm.struct<(i64)> 
    %8 = llvm.mlir.constant(8 : i64) : i64
    %9 = llvm.sdiv %7, %8  : i64
    %10 = llvm.bitcast %6 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %11 = cc.stdvec_init %10, %9 : (!llvm.ptr<f64>, i64) -> !cc.stdvec<f64>
    %12 = llvm.getelementptr %6[%7] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    call @__nvqpp__mlirgen__ansatzVec(%11) : (!cc.stdvec<f64>) -> ()
    %nil = call @__nvqpp_zeroDynamicResult() : () -> !llvm.struct<(ptr<i8>, i64)>
    return %nil : !llvm.struct<(ptr<i8>, i64)>
  }
})#";

  cudaq::registry::deviceCodeHolderAdd("ansatzVec", quakeCode.data());
  std::size_t (*ptr)(void **, void **);
  ptr = ansatzArgsCreatorVector;
  cudaq::registry::cudaqRegisterArgsCreator("ansatzVec",
                                            reinterpret_cast<char *>(ptr));

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  {
    auto jobs =
        cudaq::observe_detach("ansatzVec", h, std::vector<double>{0.59});
    for (auto &j : jobs) {
      printf("%s %s\n", j.id.data(), j.name.data());
    }
    double res = cudaq::observe(h, jobs);

    EXPECT_NEAR(res, -1.74, 1e-2);
  }

  // Observe with shots
  {
    std::size_t shots = 10000;
    auto jobs =
        cudaq::observe_detach("ansatzVec", shots, h, std::vector<double>{0.59});
    for (auto &j : jobs) {
      printf("%s %s\n", j.id.data(), j.name.data());
    }
    auto res = cudaq::observe(h, jobs);

    EXPECT_NEAR(res.exp_val_z(), -1.74, 1e-1);
    res.dump();
    auto x0x1Counts = res.counts(x(0) * x(1));
    x0x1Counts.dump();
    EXPECT_TRUE(x0x1Counts.size() == 4);

    auto z1Counts = res.counts(z(1));
    z1Counts.dump();

    EXPECT_EQ(2, z1Counts.size());
  }
}
