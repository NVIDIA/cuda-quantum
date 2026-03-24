/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --kernel-execution=generate-run-stack=1 --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --decomposition=enable-patterns="CCXToCCZ,CCZToCX" --combine-quantum-alloc --canonicalize --convert-to-qir-api --return-to-output-log --symbol-dce --canonicalize | FileCheck %s
// clang-format on

#include <cudaq.h>

struct K9 {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(5);
    cudaq::qubit p;
    return mz(q);
  }
};

__qpu__ bool kernel_of_truth() { return true; }

__qpu__ int kernel_of_corn() { return 0xDeadBeef; }

class CliffDiver {
public:
  double operator()() __qpu__ { return 42.0; }
};

__qpu__ float kernel_of_wheat() { return 13.1f; }

class CliffClimber {
public:
  char operator()() __qpu__ { return 'c'; }
};

__qpu__ unsigned long long this_is_not_a_drill() { return 123400000ull; }

__qpu__ unsigned short this_is_a_hammer() { return 2387; }

struct Soap {
  bool bubble;
  int on_a_rope;
};

struct CliffHanger {
  Soap operator()() __qpu__ { return {true, 747}; }
};

__qpu__ std::vector<float> unary_test_list(int count) {
  cudaq::qvector v(count);
  std::vector<float> vec{0, 1};
  return vec;
}

__qpu__ std::vector<bool> unary_test_list2(int count) {
  cudaq::qvector v(count);
  std::vector<bool> vec{false, true};
  return vec;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__K9.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__K9() : () -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__K9.run.entry(
// CHECK:           %[[VAL_2:.*]] = constant @K9.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.addressof @K9.run.kernelName : !llvm.ptr<array<7 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_truth._Z15kernel_of_truthv.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__function_kernel_of_truth._Z15kernel_of_truthv() : () -> i1
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i1" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_0]], %[[VAL_2]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_truth._Z15kernel_of_truthv.run.entry()
// CHECK:           %[[VAL_1:.*]] = constant @function_kernel_of_truth._Z15kernel_of_truthv.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.addressof @function_kernel_of_truth._Z15kernel_of_truthv.run.kernelName : !llvm.ptr<array<50 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_corn._Z14kernel_of_cornv.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__function_kernel_of_corn._Z14kernel_of_cornv() : () -> i32
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i32" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = cc.cast signed %[[VAL_0]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_3]], %[[VAL_2]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_corn._Z14kernel_of_cornv.run.entry()
// CHECK:           %[[VAL_1:.*]] = constant @function_kernel_of_corn._Z14kernel_of_cornv.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.addressof @function_kernel_of_corn._Z14kernel_of_cornv.run.kernelName : !llvm.ptr<array<48 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffDiver.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__CliffDiver() : () -> f64
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "f64" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__double_record_output(%[[VAL_0]], %[[VAL_2]]) : (f64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffDiver.run.entry(
// CHECK:           %[[VAL_2:.*]] = constant @CliffDiver.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.addressof @CliffDiver.run.kernelName : !llvm.ptr<array<15 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_wheat._Z15kernel_of_wheatv.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__function_kernel_of_wheat._Z15kernel_of_wheatv() : () -> f32
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "f32" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_0]] : (f32) -> f64
// CHECK:           call @__quantum__rt__double_record_output(%[[VAL_3]], %[[VAL_2]]) : (f64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_wheat._Z15kernel_of_wheatv.run.entry()
// CHECK:           %[[VAL_1:.*]] = constant @function_kernel_of_wheat._Z15kernel_of_wheatv.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.addressof @function_kernel_of_wheat._Z15kernel_of_wheatv.run.kernelName : !llvm.ptr<array<50 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffClimber.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__CliffClimber() : () -> i8
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i8" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = cc.cast signed %[[VAL_0]] : (i8) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_3]], %[[VAL_2]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffClimber.run.entry(
// CHECK:           %[[VAL_2:.*]] = constant @CliffClimber.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.addressof @CliffClimber.run.kernelName : !llvm.ptr<array<17 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_this_is_not_a_drill._Z19this_is_not_a_drillv.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__function_this_is_not_a_drill._Z19this_is_not_a_drillv() : () -> i64
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i64" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_0]], %[[VAL_2]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_this_is_not_a_drill._Z19this_is_not_a_drillv.run.entry()
// CHECK:           %[[VAL_1:.*]] = constant @function_this_is_not_a_drill._Z19this_is_not_a_drillv.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.addressof @function_this_is_not_a_drill._Z19this_is_not_a_drillv.run.kernelName : !llvm.ptr<array<58 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_this_is_a_hammer._Z16this_is_a_hammerv.run()
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__function_this_is_a_hammer._Z16this_is_a_hammerv() : () -> i16
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i16" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_3:.*]] = cc.cast signed %[[VAL_0]] : (i16) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_3]], %[[VAL_2]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_this_is_a_hammer._Z16this_is_a_hammerv.run.entry()
// CHECK:           %[[VAL_1:.*]] = constant @function_this_is_a_hammer._Z16this_is_a_hammerv.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.addressof @function_this_is_a_hammer._Z16this_is_a_hammerv.run.kernelName : !llvm.ptr<array<52 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffHanger.run()
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_1:.*]] = call @__nvqpp__mlirgen__CliffHanger() : () -> !cc.struct<"Soap" {i1, i32} [64,4]>
// CHECK:           %[[VAL_2:.*]] = cc.string_literal "tuple<i1, i32>" : !cc.ptr<!cc.array<i8 x 15>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<i8 x 15>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__tuple_record_output(%[[VAL_0]], %[[VAL_3]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_4:.*]] = cc.extract_value %[[VAL_1]][0] : (!cc.struct<"Soap" {i1, i32} [64,4]>) -> i1
// CHECK:           %[[VAL_5:.*]] = cc.string_literal ".0" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_4]], %[[VAL_6]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_7:.*]] = cc.extract_value %[[VAL_1]][1] : (!cc.struct<"Soap" {i1, i32} [64,4]>) -> i32
// CHECK:           %[[VAL_8:.*]] = cc.string_literal ".1" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = cc.cast signed %[[VAL_7]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_10]], %[[VAL_9]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffHanger.run.entry(
// CHECK:           %[[VAL_2:.*]] = constant @CliffHanger.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.addressof @CliffHanger.run.kernelName : !llvm.ptr<array<16 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unary_test_list._Z15unary_test_listi.run(
// CHECK-SAME:      %[[VAL_0:.*]]: i32)
// CHECK:           %[[VAL_1:.*]] = call @__nvqpp__mlirgen__function_unary_test_list._Z15unary_test_listi(%[[VAL_0]]) : (i32) -> !cc.stdvec<f32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unary_test_list._Z15unary_test_listi.run.entry(
// CHECK:           %[[VAL_3:.*]] = constant @function_unary_test_list._Z15unary_test_listi.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.addressof @function_unary_test_list._Z15unary_test_listi.run.kernelName : !llvm.ptr<array<50 x i8>>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unary_test_list2._Z16unary_test_list2i.run(
// CHECK-SAME:      %[[VAL_0:.*]]: i32)
// CHECK:           %[[VAL_1:.*]] = call @__nvqpp__mlirgen__function_unary_test_list2._Z16unary_test_list2i(%[[VAL_0]]) : (i32) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unary_test_list2._Z16unary_test_list2i.run.entry(
// CHECK:           %[[VAL_3:.*]] = constant @function_unary_test_list2._Z16unary_test_list2i.run.thunk : (!cc.ptr<i8>, i1) -> !cc.struct<{!cc.ptr<i8>, i64}>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.addressof @function_unary_test_list2._Z16unary_test_list2i.run.kernelName : !llvm.ptr<array<52 x i8>>
