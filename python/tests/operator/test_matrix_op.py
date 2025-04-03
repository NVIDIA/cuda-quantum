# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, pytest
from cudaq.ops import * # FIXME: module name
from op_utils import * # test helpers


def test_definitions():
    dims = {0: 2, 1: 3}
    # FIXME: allow for params as kwargs
    params = {"squeezing": 0.5 + 1.2j, "displacement": 0.5 + 1.2j}
    assert np.allclose(number(1).to_matrix(dims), number_matrix(3))
    assert np.allclose(parity(1).to_matrix(dims), parity_matrix(3))
    assert np.allclose(position(1).to_matrix(dims), position_matrix(3))
    assert np.allclose(momentum(1).to_matrix(dims), momentum_matrix(3))
    assert np.allclose(squeeze(1).to_matrix(dims, params), squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(displace(1).to_matrix(dims, params), displace_matrix(3, 0.5 + 1.2j))


def test_construction():
    pass


def test_iteration():
    prod1 = position(1) * momentum(0)
    prod2 = number(0) * parity(0)
    sum = prod1 + prod2
    for p1, p2 in zip(sum, [prod1, prod2]):
        for t1, t2 in zip(p1, p2):
            assert t1 == t2
    sum_terms = 0
    prod_terms = 0
    for prod in sum:
        sum_terms += 1
        for t in prod:
            prod_terms += 1
    assert sum_terms == 2
    assert prod_terms == 4


def test_properties():

    prod1 = position(1) * momentum(0)
    prod2 = number(1) * parity(3)
    sum = prod1 + prod2
    assert prod1.degrees() == [0, 1]
    assert prod2.degrees() == [1, 3]
    assert sum.degrees() == [0, 1, 3]
    assert prod1.min_degree() == 0
    assert prod1.max_degree() == 1
    assert prod2.min_degree() == 1
    assert prod2.max_degree() == 3
    assert sum.min_degree() == 0
    assert sum.max_degree() == 3

    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    assert sum.num_terms() == 2
    sum += prod1
    assert sum.num_terms() == 2
    prod1_mat = np.kron(identity_matrix(4), np.kron(position_matrix(3), momentum_matrix(2)))
    prod2_mat = np.kron(parity_matrix(4), np.kron(number_matrix(3), identity_matrix(2)))
    assert np.allclose(sum.to_matrix(dims), prod1_mat + prod1_mat + prod2_mat)

    prod1.dump()
    sum.dump()
    assert prod1.to_string() == "(1.000000+0.000000i) * momentum(0)position(1)"
    assert sum.to_string() == "(2.000000+0.000000i) * momentum(0)position(1) + (1.000000+0.000000i) * number(1)parity(3)"


def test_canonicalization():
    pass


def test_equality():
    pass


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])