# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP %s

from cudaq.mlir.ir import Context, Module
from cudaq.mlir.passmanager import PassManager


def test_expand_mixed_sized_and_unresolved_controls():
    ir = """
module {
  func.func @mixed_controls() {
    %sized = quake.alloca !quake.veq<1>
    %count = arith.constant 16 : i64
    %unresolved = quake.alloca !quake.veq<?>[%count : i64]
    %target = quake.alloca !quake.ref
    quake.x [%sized, %unresolved] %target : (!quake.veq<1>, !quake.veq<?>, !quake.ref) -> ()
    return
  }
}
"""
    with Context():
        module = Module.parse(ir)
        pass_manager = PassManager.parse(
            "builtin.module(func.func(expand-control-veqs))")
        pass_manager.run(module.operation)

    assert "quake.extract_ref" in str(module)
    assert "!quake.ref, !quake.veq<?>, !quake.ref" in str(module)
