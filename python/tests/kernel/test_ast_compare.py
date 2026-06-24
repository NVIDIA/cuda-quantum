# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pytest
import cudaq


def cmpfop(predicate, left, right):
    operations = {
        2: lambda l, r: l > r,
        3: lambda l, r: l >= r,
        4: lambda l, r: l < r,
        5: lambda l, r: l <= r,
        1: lambda l, r: l == r,
        6: lambda l, r: l != r,
    }
    return operations[predicate](left, right)


def cmpiop(predicate, left, right):
    operations = {
        4: lambda l, r: l > r,
        5: lambda l, r: l >= r,
        2: lambda l, r: l < r,
        7: lambda l, r: l <= r,
        0: lambda l, r: l == r,
        1: lambda l, r: l != r,
    }
    return operations[predicate](left, right)


@pytest.mark.parametrize(
    "left, right, operation, expected",
    [
        # Integer comparisons
        (3, 5, "Lt", True),
        (5, 3, "Gt", True),
        (3, 3, "Eq", True),
        (3, 5, "LtE", True),
        (5, 5, "GtE", True),
        (3, 5, "NotEq", True),
        (5, 5, "NotEq", False),

        # Float comparisons
        (3.2, 4.5, "Lt", True),
        (4.5, 3.2, "Gt", True),
        (3.2, 3.2, "Eq", True),
        (3.2, 4.5, "LtE", True),
        (4.5, 4.5, "GtE", True),
        (3.2, 4.5, "NotEq", True),
        (4.5, 4.5, "NotEq", False),

        # Mixed comparisons
        (3, 4.5, "Lt", True),
        (4.5, 3, "Gt", True),
        (3, 3.0, "Eq", True),
        (3, 4.5, "LtE", True),
        (4.5, 4, "GtE", True),
        (3, 4.5, "NotEq", True),
    ],
)
def test_visit_compare(left, right, operation, expected):
    result = None

    if operation in ["Gt", "GtE", "Lt", "LtE", "Eq", "NotEq"]:
        if isinstance(left, float) or isinstance(right, float):
            predicate = {
                "Gt": 2,
                "GtE": 3,
                "Lt": 4,
                "LtE": 5,
                "Eq": 1,
                "NotEq": 6,
            }[operation]
            result = cmpfop(predicate, left, right)
        else:
            predicate = {
                "Gt": 4,
                "GtE": 5,
                "Lt": 2,
                "LtE": 7,
                "Eq": 0,
                "NotEq": 1,
            }[operation]
            result = cmpiop(predicate, left, right)

    assert result == expected


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
