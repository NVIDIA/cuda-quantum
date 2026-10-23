# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonical QEC manifest values preserve their types and validation."""

import json
from enum import IntEnum

import pytest

from cudaq.logical.qec.lowering import _manifest_value


@pytest.mark.parametrize("sequence", (list, tuple))
@pytest.mark.parametrize("entries", ([], [0, 1, -3, 2**100, True, False]))
def test_integer_sequences_preserve_canonical_json(sequence, entries):
    value = sequence(entries)
    result = _manifest_value(value)
    expected = {"kind": sequence.__name__, "items": entries}
    assert json.dumps(result) == json.dumps(expected)
    assert result["items"] is not value


def test_nested_manifest_values_preserve_container_and_scalar_types():

    class Bit(IntEnum):
        ZERO = 0
        ONE = 1

    value = ((Bit.ZERO, Bit.ONE, True), [0, 1], (0.0, None, "1"))
    assert json.dumps(_manifest_value(value)) == json.dumps({
        "kind":
            "tuple",
        "items": [
            {
                "kind": "tuple",
                "items": [0, 1, True]
            },
            {
                "kind": "list",
                "items": [0, 1]
            },
            {
                "kind": "tuple",
                "items": [{
                    "kind": "float",
                    "value": "0.0"
                }, None, "1"],
            },
        ],
    })


def test_manifest_sequence_subclasses_are_traversed_once():

    class SinglePassList(list):

        def __iter__(self):
            assert not getattr(self, "visited", False)
            self.visited = True
            return super().__iter__()

    assert _manifest_value(SinglePassList([0, 1])) == {
        "kind": "list",
        "items": [0, 1]
    }


def test_manifest_rows_are_copied_without_caching_mutable_input():
    row = [0, 1]
    original = _manifest_value(row)
    row[0] = 1
    assert original == {"kind": "list", "items": [0, 1]}
    assert _manifest_value(row) == {"kind": "list", "items": [1, 1]}


@pytest.mark.parametrize("value, message", (
    (float("nan"), "finite floats"),
    (float("inf"), "finite floats"),
    (object(), "canonical typed values"),
))
def test_mixed_manifest_sequences_still_validate_every_entry(value, message):
    active = set()
    with pytest.raises(TypeError, match=message):
        _manifest_value(([0, 1], [1, value]), active)
    assert not active


def test_manifest_cycles_are_rejected_after_integer_rows():
    value = [[0, 1]]
    value.append(value)
    active = set()
    with pytest.raises(ValueError, match="acyclic"):
        _manifest_value(value, active)
    assert not active
