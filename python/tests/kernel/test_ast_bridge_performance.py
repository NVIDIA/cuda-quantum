# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import importlib

import cudaq

ast_bridge = importlib.import_module("cudaq.kernel.ast_bridge")


def test_repeated_qualified_calls_reuse_name_resolution(monkeypatch):

    @cudaq.kernel
    def repeated_controls():
        q = cudaq.qvector(2)
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])

    @cudaq.kernel
    def repeated_controls_in_another_bridge():
        q = cudaq.qvector(2)
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[1])

    original_stack = ast_bridge.inspect.stack
    original_recover = ast_bridge.recover_value_of_or_none
    stack_calls = 0
    qualified_recovery_calls = 0

    def counting_stack(*args, **kwargs):
        nonlocal stack_calls
        stack_calls += 1
        return original_stack(*args, **kwargs)

    def counting_recover(name, *args, **kwargs):
        nonlocal qualified_recovery_calls
        if name == "x.ctrl":
            qualified_recovery_calls += 1
        return original_recover(name, *args, **kwargs)

    monkeypatch.setattr(ast_bridge.inspect, "stack", counting_stack)
    monkeypatch.setattr(ast_bridge, "recover_value_of_or_none",
                        counting_recover)
    repeated_controls.compile()

    # resolveQualifiedName is the only expected inspect.stack caller on this
    # compile path, so bound all stack inspections performed by the bridge.
    assert stack_calls <= 1
    assert qualified_recovery_calls == 1

    repeated_controls_in_another_bridge.compile()

    assert stack_calls <= 2
    assert qualified_recovery_calls == 2
