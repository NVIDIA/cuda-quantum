# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq


def count_extract_refs(kernel):
    kernel.compile()
    return str(kernel.qkeModule).count("quake.extract_ref")


def test_reuses_repeated_fixed_qrefs_by_vector_identity():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        alias = q
        other = cudaq.qvector(1)
        x(q[0])
        x(alias[0])
        x(q[0])
        x(alias[0])
        x(other[0])
        x(other[0])

    assert count_extract_refs(kernel) == 2
    assert cudaq.sample(kernel).count("000") == 1000


def test_fixed_qref_reuse_respects_structured_dominance():

    @cudaq.kernel
    def outer_definition_dominates(condition: bool):
        q = cudaq.qvector(1)
        x(q[0])
        if condition:
            x(q[0])
        else:
            y(q[0])
        x(q[0])

    @cudaq.kernel
    def branch_definitions_do_not_escape(condition: bool):
        q = cudaq.qvector(1)
        if condition:
            x(q[0])
            x(q[0])
        else:
            y(q[0])
            y(q[0])
        z(q[0])

    assert count_extract_refs(outer_definition_dominates) == 1
    assert count_extract_refs(branch_definitions_do_not_escape) == 3


def test_fixed_qref_reuse_respects_loop_dominance():

    @cudaq.kernel
    def outer_definition_dominates(count: int):
        q = cudaq.qvector(1)
        x(q[0])
        for _ in range(count):
            x(q[0])
        x(q[0])

    @cudaq.kernel
    def loop_definition_does_not_escape(count: int):
        q = cudaq.qvector(1)
        for _ in range(count):
            x(q[0])
            x(q[0])
        z(q[0])

    @cudaq.kernel
    def while_condition_definition_does_not_escape():
        q = cudaq.qvector(1)
        while mz(q[0]):
            x(q[0])
        x(q[0])

    assert count_extract_refs(outer_definition_dominates) == 1
    assert count_extract_refs(loop_definition_does_not_escape) == 2
    assert count_extract_refs(while_condition_definition_does_not_escape) == 3


def test_dynamic_and_negative_qrefs_are_not_reused():

    @cudaq.kernel
    def dynamic_index(index: int):
        q = cudaq.qvector(2)
        x(q[index])
        x(q[index])

    @cudaq.kernel
    def negative_index():
        q = cudaq.qvector(2)
        x(q[-1])
        x(q[-1])

    assert count_extract_refs(dynamic_index) == 2
    assert count_extract_refs(negative_index) == 2
    assert cudaq.sample(dynamic_index, 0).count("00") == 1000
    assert cudaq.sample(dynamic_index, 1).count("00") == 1000
    assert cudaq.sample(negative_index).count("00") == 1000


def test_repeated_out_of_bounds_fixed_qrefs_still_diagnose(capfd):

    @cudaq.kernel
    def direct():
        q = cudaq.qvector(2)
        x(q[2])
        x(q[2])

    @cudaq.kernel
    def through_alias():
        q = cudaq.qvector(2)
        alias = q
        x(alias[2])
        x(alias[2])

    with pytest.raises(RuntimeError, match="could not compile code"):
        direct.compile()
    assert "invalid index [2] because >= size [2]" in capfd.readouterr().err

    with pytest.raises(RuntimeError, match="could not compile code"):
        through_alias.compile()
    assert "invalid index [2] because >= size [2]" in capfd.readouterr().err
