/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake --cudaq-run=1 %cpp_std %s | cudaq-opt --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --decomposition=enable-patterns="CCXToCCZ,CCZToCX" --combine-quantum-alloc --canonicalize --convert-to-qir-api --return-to-output-log --symbol-dce --canonicalize | FileCheck %s
// clang-format on

#include <cudaq.h>

struct K9 {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(5);
    cudaq::qubit p;
    return mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__K9() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel", "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 6 : i64
// CHECK:           %[[VAL_7:.*]] = call @__quantum__rt__qubit_allocate_array(%[[VAL_6]]) : (i64) -> !cc.ptr<!llvm.struct<"Array", opaque>>
// CHECK:           %[[VAL_8:.*]] = cc.alloca !cc.array<i8 x 5>
// CHECK:           %[[VAL_9:.*]] = call @__quantum__rt__array_get_element_ptr_1d(%[[VAL_7]], %[[VAL_3]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>, i64) -> !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_11:.*]] = call @__quantum__qis__mz(%[[VAL_10]]) {registerName = "r00000"} : (!cc.ptr<!llvm.struct<"Qubit", opaque>>) -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_13:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i1>
// CHECK:           %[[VAL_14:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<i8 x 5>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = cc.cast unsigned %[[VAL_13]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_15]], %[[VAL_14]] : !cc.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = call @__quantum__rt__array_get_element_ptr_1d(%[[VAL_7]], %[[VAL_4]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>, i64) -> !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_17:.*]] = cc.load %[[VAL_16]] : !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_18:.*]] = call @__quantum__qis__mz(%[[VAL_17]]) {registerName = "r00001"} : (!cc.ptr<!llvm.struct<"Qubit", opaque>>) -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:           %[[VAL_19:.*]] = cc.cast %[[VAL_18]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i1>
// CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<i8 x 5>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_22:.*]] = cc.cast unsigned %[[VAL_20]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_22]], %[[VAL_21]] : !cc.ptr<i8>
// CHECK:           %[[VAL_23:.*]] = call @__quantum__rt__array_get_element_ptr_1d(%[[VAL_7]], %[[VAL_2]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>, i64) -> !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_25:.*]] = call @__quantum__qis__mz(%[[VAL_24]]) {registerName = "r00002"} : (!cc.ptr<!llvm.struct<"Qubit", opaque>>) -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:           %[[VAL_26:.*]] = cc.cast %[[VAL_25]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_27:.*]] = cc.load %[[VAL_26]] : !cc.ptr<i1>
// CHECK:           %[[VAL_28:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<i8 x 5>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_29:.*]] = cc.cast unsigned %[[VAL_27]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_29]], %[[VAL_28]] : !cc.ptr<i8>
// CHECK:           %[[VAL_30:.*]] = call @__quantum__rt__array_get_element_ptr_1d(%[[VAL_7]], %[[VAL_1]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>, i64) -> !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_31:.*]] = cc.load %[[VAL_30]] : !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_32:.*]] = call @__quantum__qis__mz(%[[VAL_31]]) {registerName = "r00003"} : (!cc.ptr<!llvm.struct<"Qubit", opaque>>) -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:           %[[VAL_33:.*]] = cc.cast %[[VAL_32]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_34:.*]] = cc.load %[[VAL_33]] : !cc.ptr<i1>
// CHECK:           %[[VAL_35:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<i8 x 5>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_36:.*]] = cc.cast unsigned %[[VAL_34]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_36]], %[[VAL_35]] : !cc.ptr<i8>
// CHECK:           %[[VAL_37:.*]] = call @__quantum__rt__array_get_element_ptr_1d(%[[VAL_7]], %[[VAL_0]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>, i64) -> !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_38:.*]] = cc.load %[[VAL_37]] : !cc.ptr<!cc.ptr<!llvm.struct<"Qubit", opaque>>>
// CHECK:           %[[VAL_39:.*]] = call @__quantum__qis__mz(%[[VAL_38]]) {registerName = "r00004"} : (!cc.ptr<!llvm.struct<"Qubit", opaque>>) -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:           %[[VAL_40:.*]] = cc.cast %[[VAL_39]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_41:.*]] = cc.load %[[VAL_40]] : !cc.ptr<i1>
// CHECK:           %[[VAL_42:.*]] = cc.compute_ptr %[[VAL_8]][4] : (!cc.ptr<!cc.array<i8 x 5>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_43:.*]] = cc.cast unsigned %[[VAL_41]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_43]], %[[VAL_42]] : !cc.ptr<i8>
// CHECK:           %[[VAL_44:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_14]], %[[VAL_5]], %[[VAL_4]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_45:.*]] = cc.stdvec_init %[[VAL_44]], %[[VAL_5]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           call @__quantum__rt__qubit_release_array(%[[VAL_7]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>) -> ()
// CHECK:           %[[VAL_46:.*]] = cc.string_literal "array<i1 x 5>" : !cc.ptr<!cc.array<i8 x 14>>
// CHECK:           %[[VAL_47:.*]] = cc.cast %[[VAL_46]] : (!cc.ptr<!cc.array<i8 x 14>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__array_record_output(%[[VAL_5]], %[[VAL_47]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_48:.*]] = cc.cast %[[VAL_44]] : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<i1 x ?>>
// CHECK:           %[[VAL_49:.*]] = cc.cast %[[VAL_44]] : (!cc.ptr<i8>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_50:.*]] = cc.load %[[VAL_49]] : !cc.ptr<i1>
// CHECK:           %[[VAL_51:.*]] = cc.string_literal "[0]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_52:.*]] = cc.cast %[[VAL_51]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_50]], %[[VAL_52]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_53:.*]] = cc.compute_ptr %[[VAL_48]][1] : (!cc.ptr<!cc.array<i1 x ?>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_54:.*]] = cc.load %[[VAL_53]] : !cc.ptr<i1>
// CHECK:           %[[VAL_55:.*]] = cc.string_literal "[1]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_56:.*]] = cc.cast %[[VAL_55]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_54]], %[[VAL_56]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_57:.*]] = cc.compute_ptr %[[VAL_48]][2] : (!cc.ptr<!cc.array<i1 x ?>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_58:.*]] = cc.load %[[VAL_57]] : !cc.ptr<i1>
// CHECK:           %[[VAL_59:.*]] = cc.string_literal "[2]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_60:.*]] = cc.cast %[[VAL_59]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_58]], %[[VAL_60]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_61:.*]] = cc.compute_ptr %[[VAL_48]][3] : (!cc.ptr<!cc.array<i1 x ?>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_62:.*]] = cc.load %[[VAL_61]] : !cc.ptr<i1>
// CHECK:           %[[VAL_63:.*]] = cc.string_literal "[3]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_64:.*]] = cc.cast %[[VAL_63]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_62]], %[[VAL_64]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_65:.*]] = cc.compute_ptr %[[VAL_48]][4] : (!cc.ptr<!cc.array<i1 x ?>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_66:.*]] = cc.load %[[VAL_65]] : !cc.ptr<i1>
// CHECK:           %[[VAL_67:.*]] = cc.string_literal "[4]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_68:.*]] = cc.cast %[[VAL_67]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_66]], %[[VAL_68]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_45]] : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

__qpu__ bool kernel_of_truth() { return true; }

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_truth._Z15kernel_of_truthv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i1" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_0]], %[[VAL_2]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_0]] : i1
// CHECK:         }
// clang-format on

__qpu__ int kernel_of_corn() { return 0xDeadBeef; }

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_corn._Z14kernel_of_cornv() -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant -559038737 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant -559038737 : i32
// CHECK:           %[[VAL_2:.*]] = cc.string_literal "i32" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_0]], %[[VAL_3]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_1]] : i32
// CHECK:         }
// clang-format on

class CliffDiver {
public:
  double operator()() __qpu__ { return 42.0; }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffDiver() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel", "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4.200000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "f64" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__double_record_output(%[[VAL_0]], %[[VAL_2]]) : (f64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_0]] : f64
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_of_wheat._Z15kernel_of_wheatv() -> f32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 13.100000381469727 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.310000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = cc.string_literal "f32" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__double_record_output(%[[VAL_0]], %[[VAL_3]]) : (f64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_1]] : f32
// CHECK:         }
// clang-format on

__qpu__ float kernel_of_wheat() { return 13.1f; }

class CliffClimber {
public:
  char operator()() __qpu__ { return 'c'; }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffClimber() -> i8 attributes {"cudaq-entrypoint", "cudaq-kernel", "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 99 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 99 : i8
// CHECK:           %[[VAL_2:.*]] = cc.string_literal "i8" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_0]], %[[VAL_3]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_1]] : i8
// CHECK:         }
// clang-format on

__qpu__ unsigned long long this_is_not_a_drill() { return 123400000ull; }

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_this_is_not_a_drill._Z19this_is_not_a_drillv() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 123400000 : i64
// CHECK:           %[[VAL_1:.*]] = cc.string_literal "i64" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_0]], %[[VAL_2]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_0]] : i64
// CHECK:         }
// clang-format on

__qpu__ unsigned short this_is_a_hammer() { return 2387; }

struct Soap {
  bool bubble;
  int on_a_rope;
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_this_is_a_hammer._Z16this_is_a_hammerv() -> i16 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2387 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2387 : i16
// CHECK:           %[[VAL_2:.*]] = cc.string_literal "i16" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_0]], %[[VAL_3]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_1]] : i16
// CHECK:         }
// clang-format on

struct CliffHanger {
  Soap operator()() __qpu__ { return {true, 747}; }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CliffHanger() -> !cc.struct<"Soap" {i1, i32} [64,4]> attributes {"cudaq-entrypoint", "cudaq-kernel", "qir-api"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 747 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<"Soap" {i1, i32} [64,4]>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.struct<"Soap" {i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.struct<"Soap" {i1, i32} [64,4]>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<!cc.struct<"Soap" {i1, i32} [64,4]>>
// CHECK:           %[[VAL_7:.*]] = cc.string_literal "tuple<i1, i32>" : !cc.ptr<!cc.array<i8 x 15>>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<i8 x 15>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__tuple_record_output(%[[VAL_0]], %[[VAL_8]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_9:.*]] = cc.extract_value %[[VAL_6]][0] : (!cc.struct<"Soap" {i1, i32} [64,4]>) -> i1
// CHECK:           %[[VAL_10:.*]] = cc.string_literal ".0" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_10]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_9]], %[[VAL_11]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_12:.*]] = cc.extract_value %[[VAL_6]][1] : (!cc.struct<"Soap" {i1, i32} [64,4]>) -> i32
// CHECK:           %[[VAL_13:.*]] = cc.string_literal ".1" : !cc.ptr<!cc.array<i8 x 3>>
// CHECK:           %[[VAL_14:.*]] = cc.cast %[[VAL_13]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = cc.cast signed %[[VAL_12]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_15]], %[[VAL_14]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_6]] : !cc.struct<"Soap" {i1, i32} [64,4]>
// CHECK:         }
// clang-format on

__qpu__ std::vector<float> unary_test_list(int count) {
 cudaq::qvector v(count);
 std::vector<float> vec {0, 1};
 return vec;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unary_test_list._Z15unary_test_listi(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) -> !cc.stdvec<f32> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = call @__quantum__rt__qubit_allocate_array(%[[VAL_7]]) : (i64) -> !cc.ptr<!llvm.struct<"Array", opaque>>
// CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.array<f32 x 2>
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_9]] : (!cc.ptr<!cc.array<f32 x 2>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<f32>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_9]][1] : (!cc.ptr<!cc.array<f32 x 2>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<f32>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_9]] : (!cc.ptr<!cc.array<f32 x 2>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_12]], %[[VAL_4]], %[[VAL_3]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = cc.stdvec_init %[[VAL_13]], %[[VAL_4]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<f32>
// CHECK:           call @__quantum__rt__qubit_release_array(%[[VAL_8]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>) -> ()
// CHECK:           %[[VAL_15:.*]] = cc.string_literal "array<f32 x 2>" : !cc.ptr<!cc.array<i8 x 15>>
// CHECK:           %[[VAL_16:.*]] = cc.cast %[[VAL_15]] : (!cc.ptr<!cc.array<i8 x 15>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__array_record_output(%[[VAL_4]], %[[VAL_16]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_17:.*]] = cc.cast %[[VAL_13]] : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<f32 x ?>>
// CHECK:           %[[VAL_18:.*]] = cc.cast %[[VAL_13]] : (!cc.ptr<i8>) -> !cc.ptr<f32>
// CHECK:           %[[VAL_19:.*]] = cc.load %[[VAL_18]] : !cc.ptr<f32>
// CHECK:           %[[VAL_20:.*]] = cc.string_literal "[0]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_21:.*]] = cc.cast %[[VAL_20]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_22:.*]] = cc.cast %[[VAL_19]] : (f32) -> f64
// CHECK:           call @__quantum__rt__double_record_output(%[[VAL_22]], %[[VAL_21]]) : (f64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_23:.*]] = cc.compute_ptr %[[VAL_17]][1] : (!cc.ptr<!cc.array<f32 x ?>>) -> !cc.ptr<f32>
// CHECK:           %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<f32>
// CHECK:           %[[VAL_25:.*]] = cc.string_literal "[1]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_26:.*]] = cc.cast %[[VAL_25]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_27:.*]] = cc.cast %[[VAL_24]] : (f32) -> f64
// CHECK:           call @__quantum__rt__double_record_output(%[[VAL_27]], %[[VAL_26]]) : (f64, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_14]] : !cc.stdvec<f32>
// CHECK:         }
// clang-format on

__qpu__ std::vector<bool> unary_test_list2(int count) {
 cudaq::qvector v(count);
 std::vector<bool> vec {false, true};
 return vec;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unary_test_list2._Z16unary_test_list2i(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, "qir-api"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = call @__quantum__rt__qubit_allocate_array(%[[VAL_7]]) : (i64) -> !cc.ptr<!llvm.struct<"Array", opaque>>
// CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.array<i1 x 2>
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_9]] : (!cc.ptr<!cc.array<i1 x 2>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_10]] : !cc.ptr<i1>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_9]][1] : (!cc.ptr<!cc.array<i1 x 2>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_11]] : !cc.ptr<i1>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_9]] : (!cc.ptr<!cc.array<i1 x 2>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_12]], %[[VAL_2]], %[[VAL_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = cc.stdvec_init %[[VAL_13]], %[[VAL_2]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           call @__quantum__rt__qubit_release_array(%[[VAL_8]]) : (!cc.ptr<!llvm.struct<"Array", opaque>>) -> ()
// CHECK:           %[[VAL_15:.*]] = cc.string_literal "array<i1 x 2>" : !cc.ptr<!cc.array<i8 x 14>>
// CHECK:           %[[VAL_16:.*]] = cc.cast %[[VAL_15]] : (!cc.ptr<!cc.array<i8 x 14>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__array_record_output(%[[VAL_2]], %[[VAL_16]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_17:.*]] = cc.cast %[[VAL_13]] : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<i1 x ?>>
// CHECK:           %[[VAL_18:.*]] = cc.cast %[[VAL_13]] : (!cc.ptr<i8>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_19:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i1>
// CHECK:           %[[VAL_20:.*]] = cc.string_literal "[0]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_21:.*]] = cc.cast %[[VAL_20]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_19]], %[[VAL_21]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_17]][1] : (!cc.ptr<!cc.array<i1 x ?>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_23:.*]] = cc.load %[[VAL_22]] : !cc.ptr<i1>
// CHECK:           %[[VAL_24:.*]] = cc.string_literal "[1]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_25:.*]] = cc.cast %[[VAL_24]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__bool_record_output(%[[VAL_23]], %[[VAL_25]]) : (i1, !cc.ptr<i8>) -> ()
// CHECK:           return {cc.cudaq.run} %[[VAL_14]] : !cc.stdvec<i1>
// CHECK:         }
// clang-format on
