# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest

import cudaq

def test_custom_quantum_type():
    from dataclasses import dataclass
    @dataclass
    class patch:
        data : cudaq.qview 
        ancx : cudaq.qview 
        ancz : cudaq.qview 
    
    @cudaq.kernel
    def logicalH(p : patch):
        h(p.data)
    print(logicalH)
    
    @cudaq.kernel 
    def logicalX(p : patch):
        x(p.ancx)
    
    @cudaq.kernel 
    def logicalZ(p : patch):
        z(p.ancz)
    
    @cudaq.kernel
    def run():
        q = cudaq.qvector(2)
        r = cudaq.qvector(2)
        s = cudaq.qvector(2)
        p = patch(q, r, s)

        logicalH(p)
        logicalX(p)
        logicalZ(p)
    
    # Test here is that it compiles and runs successfully
    print(run)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__logicalH(
# CHECK-SAME:                                          %[[VAL_0:.*]]: !cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = cc.extract_value %[[VAL_0]][0] : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>) -> !quake.veq<?>
# CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
# CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_8]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__run() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = quake.relax_size %[[VAL_2]] : (!quake.veq<2>) -> !quake.veq<?>
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_5:.*]] = quake.relax_size %[[VAL_4]] : (!quake.veq<2>) -> !quake.veq<?>
# CHECK:           %[[VAL_6:.*]] = cc.undef !cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>
# CHECK:           %[[VAL_7:.*]] = cc.insert_value %[[VAL_1]], %[[VAL_6]][0] : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>, !quake.veq<?>) -> !cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>
# CHECK:           %[[VAL_8:.*]] = cc.insert_value %[[VAL_3]], %[[VAL_7]][1] : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>, !quake.veq<?>) -> !cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>
# CHECK:           %[[VAL_9:.*]] = cc.insert_value %[[VAL_5]], %[[VAL_8]][2] : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>, !quake.veq<?>) -> !cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>
# CHECK:           call @__nvqpp__mlirgen__logicalH(%[[VAL_9]]) : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>) -> ()
# CHECK:           call @__nvqpp__mlirgen__logicalX(%[[VAL_9]]) : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>) -> ()
# CHECK:           call @__nvqpp__mlirgen__logicalZ(%[[VAL_9]]) : (!cc.struct<"patch" {!quake.veq<?>, !quake.veq<?>, !quake.veq<?>}>) -> ()
# CHECK:           return
# CHECK:         }