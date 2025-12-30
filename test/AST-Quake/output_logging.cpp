/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// clang-format off
// Use a subset of the nvq++ passes related to kernel execution.
// RUN: cudaq-quake %s | cudaq-opt --kernel-execution=generate-run-stack=1 --indirect-to-direct-calls --inline  --return-to-output-log | FileCheck %s
// clang-format on

#include <cudaq.h>

std::vector<int> kernel_that_return_unknown_size_vector(std::size_t N) __qpu__ {
  std::vector<int> result(N);
  return result;
}

// Check the dynamic label generation for a kernel that returns
// a vector of unknown size with `__nvqpp_internal_tostring` and string
// concatenation with `llvm.memcpy`.
// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_that_return_unknown_size_vector._Z38kernel_that_return_unknown_size_vectorm.run(
// CHECK-SAME:                                                                                                                                 %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 32 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 14 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 12 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant false
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_9:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_10:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_10]] : !cc.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i64>
// CHECK:           %[[VAL_12:.*]] = cc.alloca i32{{\[}}%[[VAL_11]] : i64]
// CHECK:           %[[VAL_13:.*]] = cc.cast %[[VAL_12]] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_11]], %[[VAL_9]] : i64
// CHECK:           %[[VAL_15:.*]] = call @malloc(%[[VAL_14]]) : (i64) -> !cc.ptr<i8>
// CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_15]], %[[VAL_13]], %[[VAL_14]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:           %[[VAL_16:.*]] = cc.stdvec_init %[[VAL_15]], %[[VAL_11]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i32>
// CHECK:           %[[VAL_17:.*]] = cc.stdvec_size %[[VAL_16]] : (!cc.stdvec<i32>) -> i64
// CHECK:           %[[VAL_18:.*]] = call @__nvqpp_internal_number_of_digits(%[[VAL_17]]) : (i64) -> i64
// CHECK:           %[[VAL_19:.*]] = cc.alloca i8{{\[}}%[[VAL_2]] : i64]
// CHECK:           call @__nvqpp_internal_tostring(%[[VAL_19]], %[[VAL_17]]) : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> ()
// CHECK:           %[[VAL_20:.*]] = cc.cast %[[VAL_19]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_21:.*]] = cc.string_literal "array<i32 x " : !cc.ptr<!cc.array<i8 x 13>>
// CHECK:           %[[VAL_22:.*]] = cc.cast %[[VAL_21]] : (!cc.ptr<!cc.array<i8 x 13>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_23:.*]] = cc.string_literal ">" : !cc.ptr<!cc.array<i8 x 2>>
// CHECK:           %[[VAL_24:.*]] = cc.cast %[[VAL_23]] : (!cc.ptr<!cc.array<i8 x 2>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_26:.*]] = cc.alloca i8{{\[}}%[[VAL_25]] : i64]
// CHECK:           %[[VAL_27:.*]] = cc.cast %[[VAL_26]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_27]], %[[VAL_22]], %[[VAL_4]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:           %[[VAL_28:.*]] = cc.compute_ptr %[[VAL_26]][12] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_28]], %[[VAL_20]], %[[VAL_18]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_18]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_30:.*]] = cc.compute_ptr %[[VAL_26]]{{\[}}%[[VAL_29]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
// CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_30]], %[[VAL_24]], %[[VAL_6]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:           call @__quantum__rt__array_record_output(%[[VAL_17]], %[[VAL_27]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_31:.*]] = cc.stdvec_data %[[VAL_16]] : (!cc.stdvec<i32>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:           %[[VAL_32:.*]] = cc.loop while ((%[[VAL_33:.*]] = %[[VAL_7]]) -> (i64)) {
// CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_17]] : i64
// CHECK:             cc.condition %[[VAL_34]](%[[VAL_33]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_35:.*]]: i64):
// CHECK:             %[[VAL_36:.*]] = cc.compute_ptr %[[VAL_31]]{{\[}}%[[VAL_35]]] : (!cc.ptr<!cc.array<i32 x ?>>, i64) -> !cc.ptr<i32>
// CHECK:             %[[VAL_37:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:             %[[VAL_38:.*]] = func.call @__nvqpp_internal_number_of_digits(%[[VAL_35]]) : (i64) -> i64
// CHECK:             %[[VAL_39:.*]] = cc.alloca i8{{\[}}%[[VAL_2]] : i64]
// CHECK:             func.call @__nvqpp_internal_tostring(%[[VAL_39]], %[[VAL_35]]) : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> ()
// CHECK:             %[[VAL_40:.*]] = cc.cast %[[VAL_39]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:             %[[VAL_41:.*]] = cc.string_literal "[" : !cc.ptr<!cc.array<i8 x 2>>
// CHECK:             %[[VAL_42:.*]] = cc.cast %[[VAL_41]] : (!cc.ptr<!cc.array<i8 x 2>>) -> !cc.ptr<i8>
// CHECK:             %[[VAL_43:.*]] = cc.string_literal "]" : !cc.ptr<!cc.array<i8 x 2>>
// CHECK:             %[[VAL_44:.*]] = cc.cast %[[VAL_43]] : (!cc.ptr<!cc.array<i8 x 2>>) -> !cc.ptr<i8>
// CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_38]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_46:.*]] = cc.alloca i8{{\[}}%[[VAL_45]] : i64]
// CHECK:             %[[VAL_47:.*]] = cc.cast %[[VAL_46]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:             func.call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_47]], %[[VAL_42]], %[[VAL_8]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:             %[[VAL_48:.*]] = cc.compute_ptr %[[VAL_46]][1] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
// CHECK:             func.call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_48]], %[[VAL_40]], %[[VAL_38]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:             %[[VAL_49:.*]] = arith.addi %[[VAL_38]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_50:.*]] = cc.compute_ptr %[[VAL_46]]{{\[}}%[[VAL_49]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
// CHECK:             func.call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_50]], %[[VAL_44]], %[[VAL_6]], %[[VAL_5]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:             %[[VAL_51:.*]] = cc.cast signed %[[VAL_37]] : (i32) -> i64
// CHECK:             func.call @__quantum__rt__int_record_output(%[[VAL_51]], %[[VAL_47]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:             cc.continue %[[VAL_35]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_52:.*]]: i64):
// CHECK:             %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_8]] : i64
// CHECK:             cc.continue %[[VAL_53]] : i64
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }
// clang-format on

std::vector<int> kernel_that_return_known_size_vector() __qpu__ {
  std::vector<int> result(5);
  return result;
}

// If the vector size is known, all the labels are created statically with
// `cc.string_literal`.
// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_that_return_known_size_vector._Z36kernel_that_return_known_size_vectorv.run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 20 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<i32 x 5>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 5>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = call @malloc(%[[VAL_1]]) : (i64) -> !cc.ptr<i8>
// CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_5]], %[[VAL_4]], %[[VAL_1]], %[[VAL_2]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
// CHECK:           %[[VAL_6:.*]] = cc.string_literal "array<i32 x 5>" : !cc.ptr<!cc.array<i8 x 15>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<i8 x 15>>) -> !cc.ptr<i8>
// CHECK:           call @__quantum__rt__array_record_output(%[[VAL_0]], %[[VAL_7]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<i8>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_8]][0] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.string_literal "[0]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = cc.cast signed %[[VAL_10]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_13]], %[[VAL_12]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i32>
// CHECK:           %[[VAL_16:.*]] = cc.string_literal "[1]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_17:.*]] = cc.cast %[[VAL_16]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_18:.*]] = cc.cast signed %[[VAL_15]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_18]], %[[VAL_17]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i32>
// CHECK:           %[[VAL_21:.*]] = cc.string_literal "[2]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_22:.*]] = cc.cast %[[VAL_21]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_23:.*]] = cc.cast signed %[[VAL_20]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_23]], %[[VAL_22]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_24:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_25:.*]] = cc.load %[[VAL_24]] : !cc.ptr<i32>
// CHECK:           %[[VAL_26:.*]] = cc.string_literal "[3]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_27:.*]] = cc.cast %[[VAL_26]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_28:.*]] = cc.cast signed %[[VAL_25]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_28]], %[[VAL_27]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           %[[VAL_29:.*]] = cc.compute_ptr %[[VAL_8]][4] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_30:.*]] = cc.load %[[VAL_29]] : !cc.ptr<i32>
// CHECK:           %[[VAL_31:.*]] = cc.string_literal "[4]" : !cc.ptr<!cc.array<i8 x 4>>
// CHECK:           %[[VAL_32:.*]] = cc.cast %[[VAL_31]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_33:.*]] = cc.cast signed %[[VAL_30]] : (i32) -> i64
// CHECK:           call @__quantum__rt__int_record_output(%[[VAL_33]], %[[VAL_32]]) : (i64, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on
