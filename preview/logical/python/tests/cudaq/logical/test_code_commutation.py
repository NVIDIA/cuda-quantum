# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Regression coverage for code-family symplectic validation."""

import re

import pytest

from cudaq.logical.codes import Code


def _row(paulis):
    return (tuple(int(pauli in "XY") for pauli in paulis) +
            tuple(int(pauli in "ZY") for pauli in paulis))


@pytest.mark.parametrize("n", [3, 9, 65])
def test_commuting_subsystem_code_preserves_general_pauli_rows(n):
    stabilizers = tuple(
        _row("I" * i + "Z" + "I" * (n - i - 1)) for i in range(n - 2))
    anti = tuple(_row("I" * i + "X" + "I" * (n - i - 1)) for i in range(n - 2))
    logical = (_row("I" * (n - 2) + "XI"), _row("I" * (n - 2) + "ZI"))
    gauge = (_row("I" * (n - 1) + "Y"), _row("I" * (n - 1) + "Z"))
    code = Code(n=n,
                k=1,
                r=1,
                stabilizers=stabilizers,
                logicals=(logical,),
                gauge_pairs=(gauge,),
                anti_stabilizers=anti)

    assert code.logical_x_basis.rows == (logical[0],)
    assert code.logical_z_basis.rows == (logical[1],)
    assert code.gauge_x_basis.rows == (gauge[0],)
    assert code.gauge_z_basis.rows == (gauge[1],)
    assert code.anti_stabilizers.rows == anti
    assert code.encoding_clifford.rank == 2 * n


@pytest.mark.parametrize("n,k,r,stabilizers,logicals,gauges,anti,error", [
    (2, 0, 0, ("XI", "ZI"), (), (),
     (), "stabilizer generators must mutually commute"),
    (2, 1, 0, ("ZI",), (("XI", "ZI"),), (),
     (), "stabilizer/logical-X must form commuting families"),
    (2, 1, 0, ("ZI",), (("IX", "XI"),), (),
     (), "stabilizer/logical-Z must form commuting families"),
    (2, 2, 0, (), (("XI", "ZI"), ("ZX", "IZ")), (),
     (), "logical X must form commuting families"),
    (2, 2, 0, (), (("ZI", "XI"), ("IZ", "ZX")), (),
     (), "logical Z must form commuting families"),
    (2, 2, 0, (), (("XI", "ZZ"), ("IX", "IZ")), (),
     (), "logical X/Z must form canonical pairs"),
    (1, 1, 0, (), (("X", "X"),), (),
     (), "logical X/Z must form canonical pairs"),
    (2, 0, 2, (), (), (("XI", "ZI"), ("ZX", "IZ")),
     (), "gauge X must form commuting families"),
    (2, 0, 2, (), (), (("ZI", "XI"), ("IZ", "ZX")),
     (), "gauge Z must form commuting families"),
    (1, 0, 1, (), (), (("Y", "Y"),), (), "gauge X/Z must form canonical pairs"),
    (2, 1, 1, (), (("XI", "ZI"),), (("XI", "ZI"),),
     (), "logical-X/gauge-Z must form commuting families"),
    (1, 0, 0, ("Z",), (), (),
     ("Z",), "stabilizer/anti-stabilizer must form canonical pairs"),
    (2, 0, 0, ("ZI", "IZ"), (), (),
     ("XI", "ZX"), "anti-stabilizers must form commuting families"),
    (2, 1, 0, ("ZI",), (("IX", "IZ"),), (),
     ("XX",), "anti-stabilizer/logical-Z must form commuting families"),
])
def test_code_rejects_invalid_commutation(n, k, r, stabilizers, logicals,
                                          gauges, anti, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        Code(n=n,
             k=k,
             r=r,
             stabilizers=tuple(map(_row, stabilizers)),
             logicals=tuple(tuple(map(_row, pair)) for pair in logicals),
             gauge_pairs=tuple(tuple(map(_row, pair)) for pair in gauges),
             anti_stabilizers=tuple(map(_row, anti)))
