# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Workaround for kernels that may appear in jumbled order.
# RUN: PYTHONPATH=../../ pytest -rP  %s > %t && FileCheck %s < %t && FileCheck --check-prefix=NAUGHTY %s < %t && FileCheck --check-prefix=NICE %s < %t

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

# NAUGHTY-LABEL:   func.func @__nvqpp__mlirgen__logicalH(
# NAUGHTY-SAME:      %[[VAL_0:.*]]: !quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) {
# NAUGHTY:           %[[VAL_3:.*]] = quake.get_member %[[VAL_0]][0] : (!quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
# NAUGHTY:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
# NAUGHTY:           return
# NAUGHTY:         }

# NICE-LABEL:   func.func @__nvqpp__mlirgen__logicalX(
# NICE-SAME:      %[[VAL_0:.*]]: !quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) {
# NICE:           %[[VAL_3:.*]] = quake.get_member %[[VAL_0]][1] : (!quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
# NICE:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
# NICE:           return
# NICE:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__logicalZ(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) {
# CHECK:           %[[VAL_3:.*]] = quake.get_member %[[VAL_0]][2] : (!quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
# CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__run()
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = quake.make_struq %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!quake.veq<2>, !quake.veq<2>, !quake.veq<2>) -> !quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>
# CHECK:           call @__nvqpp__mlirgen__logicalH(%[[VAL_3]]) : (!quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> ()
# CHECK:           call @__nvqpp__mlirgen__logicalX(%[[VAL_3]]) : (!quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> ()
# CHECK:           call @__nvqpp__mlirgen__logicalZ(%[[VAL_3]]) : (!quake.struq<"patch": !quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> ()
# CHECK:           return
# CHECK:         }

