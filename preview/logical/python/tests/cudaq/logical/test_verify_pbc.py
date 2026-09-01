# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The PBC normal-form verifier (``verify_pbc``).

Certifies that a module is in Pauli-based-computation form: only prepare /
signed pi/4 ``pauli_rotation`` / ``mpp`` / discard / return / constant ops;
every rotation uses a positive pi/4 magnitude and carries its sign on the Pauli
product; rotations precede measurements; and the measured Pauli products
pairwise commute. ``to_pbc`` output must always pass.
"""
from __future__ import annotations

from typing import Tuple

import pytest

import cudaq.logical
from cudaq.logical._mlir_libs import _qlxRuntime as rt


def _pbc(gates, meas):
    nq = 1 + max([g[1] for g in gates] + [g[2] for g in gates if g[0] == "cx"] +
                 [m[1] for m in meas])

    def prog():
        q = cudaq.logical.allocate(nq, state=cudaq.logical.types.zero)
        for g in gates:
            if g[0] == "h":
                q[g[1]] = cudaq.logical.h(q[g[1]])
            elif g[0] == "cx":
                q[g[1]], q[g[2]] = cudaq.logical.cx(q[g[1]], q[g[2]])
            elif g[0] == "t":
                (q[g[1]],) = cudaq.logical.ops.rotate(
                    cudaq.logical.types.Z(q[g[1]]),
                    angle=cudaq.logical.types.pi / 4)
        return tuple(
            cudaq.logical.measure_z(q[i]) for _, i in [(b, i) for b, i in meas])

    prog.__annotations__["return"] = Tuple[tuple(bool for _ in meas)]
    prog = cudaq.logical.program(prog)
    mlir = cudaq.logical.compile(
        prog, pipeline=cudaq.logical.compiler.pipelines.logical()).to_mlir()
    return rt.synthesize_qlx(mlir,
                             1e-10), rt.to_pbc(rt.synthesize_qlx(mlir, 1e-10))


def test_to_pbc_output_verifies():
    _, pbc = _pbc([("h", 0), ("t", 0), ("cx", 0, 1)], [("Z", 0), ("Z", 1)])
    assert rt.verify_pbc(pbc) is True


def test_rejects_unlowered_gates():
    synth, _ = _pbc([("h", 0), ("t", 0)], [("Z", 0)])
    with pytest.raises(RuntimeError, match="no Clifford/gate actions"):
        rt.verify_pbc(synth)


# -- hand-built modules for each violation class ---------------------------
_HDR = ('module attributes {qlx.facets = [], qlx.ir_version = "0.4-draft", '
        'qlx.model_version = "0.3.10-proposed", qlx.profiles = ["p0"], '
        'qlx.stages = ["p0"]} {\n'
        '  qlx.program @c : () -> i1 attributes '
        '{qlx.profile = "p0", qlx.stage = "p0"} {\n')
_FOOT = "  }\n}\n"
_PREP = (
    '    %0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}'
    " : !qlx.logical_qubit\n")
_CST = "    %cst = arith.constant 0.78539816339744828 : f64\n"


def _rot(res, inp, num, den, xm, zm, *, sign=1):
    return (f"    {res} = qlx.apply #qlx.action<pauli_rotation>({inp}, %cst) "
            f"{{parameters = {{angle_pi_denom = {den} : i64, "
            f"angle_pi_numer = {num} : i64, sign = {sign} : i64, "
            f"x_mask = {xm} : i64, z_mask = {zm} : i64}}}} "
            ": (!qlx.logical_qubit, f64) -> !qlx.logical_qubit\n")


def _mpp(res, inp, xm, zm, n=1):
    ins = ", ".join(["!qlx.logical_qubit"] * n)
    outs = "!qlx.logical_qubit, " * n + "i1"
    return (
        f"    {res} = qlx.instrument #qlx.instrument<mpp>({inp}) "
        f"{{parameters = {{sign = 1 : i64, x_mask = {xm} : i64, z_mask = {zm} : i64}}}} "
        f": ({ins}) -> ({outs})\n")


def _module(body, ret):
    return _HDR + body + f"    qlx.return {ret} : i1\n" + _FOOT


def test_rejects_non_quarter_pi_rotation():
    m = _module(
        _PREP + _CST + _rot("%2", "%0", 1, 2, 0, 1) + _mpp("%3:2", "%2", 0, 1),
        "%3#1",
    )
    with pytest.raises(RuntimeError, match="not canonical signed pi/4"):
        rt.verify_pbc(m)


def test_rejects_rotation_after_measurement():
    body = (_PREP + _CST + _mpp("%2:2", "%0", 0, 1) +
            _rot("%3", "%2#0", 1, 4, 0, 1) + _mpp("%4:2", "%3", 0, 1))
    with pytest.raises(RuntimeError, match="rotations before measurements"):
        rt.verify_pbc(_module(body, "%4#1"))


def test_rejects_noncommuting_measurements():
    # ``mpp`` X0 then ``mpp`` Z0 on the same qubit -> anticommute.
    body = _PREP + _mpp("%2:2", "%0", 1, 0) + _mpp("%3:2", "%2#0", 0, 1)
    with pytest.raises(RuntimeError, match="pairwise commute"):
        rt.verify_pbc(_module(body, "%3#1"))


def test_accepts_valid_form():
    body = _PREP + _CST + _rot("%2", "%0", 1, 4, 1, 0) + _mpp(
        "%3:2", "%2", 1, 0)
    assert rt.verify_pbc(_module(body, "%3#1")) is True


def test_negative_quarter_pi_uses_the_product_sign():
    body = (_PREP + _CST + _rot("%2", "%0", 1, 4, 0, 1, sign=-1) +
            _mpp("%3:2", "%2", 0, 1))
    assert rt.verify_pbc(_module(body, "%3#1")) is True


def test_rejects_negative_exact_numerator_even_with_positive_operand():
    body = (_PREP + _CST + _rot("%2", "%0", -1, 4, 0, 1) +
            _mpp("%3:2", "%2", 0, 1))
    with pytest.raises(RuntimeError, match="canonical signed pi/4"):
        rt.verify_pbc(_module(body, "%3#1"))


def test_rejects_numeric_angle_that_disagrees_with_exact_metadata():
    wrong = "    %cst = arith.constant 0.5 : f64\n"
    body = (_PREP + wrong + _rot("%2", "%0", 1, 4, 0, 1) +
            _mpp("%3:2", "%2", 0, 1))
    with pytest.raises(RuntimeError, match="numeric angle"):
        rt.verify_pbc(_module(body, "%3#1"))


def test_rejects_noncanonical_product_sign():
    body = (_PREP + _CST + _rot("%2", "%0", 1, 4, 0, 1, sign=0) +
            _mpp("%3:2", "%2", 0, 1))
    with pytest.raises(RuntimeError, match=r"sign = \+/-1"):
        rt.verify_pbc(_module(body, "%3#1"))
