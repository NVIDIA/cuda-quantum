# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ python3 %s | FileCheck %s

from cudaq.mlir.ir import *
from cudaq.mlir.dialects import quake
from cudaq.mlir.dialects import builtin, func, arith

with Context() as ctx:
    quake.register_dialect()
    m = Module.create(loc=Location.unknown())
    with InsertionPoint(m.body), Location.unknown():
        f = func.FuncOp('main', ([], []))
        entry_block = f.add_entry_block()
        with InsertionPoint(entry_block):
            t = quake.RefType.get()
            v = quake.VeqType.get(10)
            iTy = IntegerType.get_signless(64)
            iAttr = IntegerAttr.get(iTy, 43)
            s = arith.ConstantOp(iTy, iAttr)

            qubit = quake.AllocaOp(t)
            target = quake.AllocaOp(t)

            qveq = quake.AllocaOp(v)
            dyn = quake.AllocaOp(quake.VeqType.get(), size=s)
            quake.HOp([], [], [], [qubit])
            quake.XOp([], [], [qubit], [target])
            ret = func.ReturnOp([])

    print(str(m))

# CHECK-LABEL:   func.func @main() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 43 : i64
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<10>
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.x {{\[}}%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }
