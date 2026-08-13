# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Robustness tests for `PyKernelDecorator.from_json`"""

import json

import pytest

import cudaq
from cudaq import PyKernelDecorator


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


class TestFromJsonRoundTrip:
    """`from_json(to_json(k))` must reconstruct the kernel."""

    def test_round_trip_preserves_name_and_source(self):
        rebuilt = PyKernelDecorator.from_json(bell.to_json())
        assert rebuilt.name == bell.name
        assert rebuilt.funcSrc == bell.funcSrc
        # `location` serializes as a JSON array, so compare element-wise (the
        # original is a tuple, the round-tripped value a list).
        assert list(rebuilt.location) == list(bell.location)


class TestFromJsonRejectsWrongShape:
    """Valid JSON of the wrong shape/type must raise a clean, informative
    error -- never a raw `TypeError`/`KeyError` from an unguarded subscript."""

    # Valid JSON that does not deserialize to an object (dict).
    NON_OBJECT = ['2', '[]', 'null', 'true', '1.5', '"hello"']

    @pytest.mark.parametrize('jStr', NON_OBJECT)
    def test_non_object_json_rejected(self, jStr):
        with pytest.raises(RuntimeError) as info:
            PyKernelDecorator.from_json(jStr)
        # The message must be actionable, not an opaque internal error.
        assert 'from_json' in str(info.value)

    def test_empty_object_missing_keys_rejected(self):
        with pytest.raises(RuntimeError) as info:
            PyKernelDecorator.from_json('{}')
        assert 'funcSrc' in str(info.value)

    @pytest.mark.parametrize('key', ['funcSrc', 'name', 'location'])
    def test_each_missing_required_key_rejected(self, key):
        obj = {
            'name': 'k',
            'location': ['f.py', 1],
            'funcSrc': 'def k():\n pass\n'
        }
        del obj[key]
        with pytest.raises(RuntimeError) as info:
            PyKernelDecorator.from_json(json.dumps(obj))
        assert key in str(info.value)

    def test_non_string_funcSrc_rejected(self):
        obj = {'name': 'k', 'location': None, 'funcSrc': 42}
        with pytest.raises(RuntimeError) as info:
            PyKernelDecorator.from_json(json.dumps(obj))
        assert 'funcSrc' in str(info.value)

    def test_non_string_name_rejected(self):
        obj = {
            'name': 1,
            'location': ['f.py', 1],
            'funcSrc': 'def k():\n pass\n'
        }
        with pytest.raises(RuntimeError) as info:
            PyKernelDecorator.from_json(json.dumps(obj))
        assert 'name' in str(info.value)

    def test_malformed_json_still_raises_value_error(self):
        # Genuinely malformed JSON is handled by json.loads itself.
        with pytest.raises(json.JSONDecodeError):
            PyKernelDecorator.from_json('{not valid json')


class TestFromJsonLocationField:
    """`location` is validated during `from_json` (so a wrong-typed
    value fails immediately instead of leaking a deferred `TypeError` from
    the compiler's diagnostic emitter), while a documented null location still
    compiles to a clean diagnostic."""

    # Valid-JSON `location` values that are neither null nor a [filename, lineno]
    # pair, each of which previously slipped through to a raw compile-time crash.
    BAD_LOCATIONS = [
        'oops', 5, True, ['a'], [], [1, 2], ['a', 'b'], ['a', 1, 2]
    ]

    @pytest.mark.parametrize('loc', BAD_LOCATIONS)
    def test_wrong_typed_location_rejected(self, loc):
        obj = {'name': 'k', 'location': loc, 'funcSrc': 'def k():\n pass\n'}
        with pytest.raises(RuntimeError) as info:
            PyKernelDecorator.from_json(json.dumps(obj))
        assert 'location' in str(info.value)

    @pytest.mark.parametrize('loc', [None, ['f.py', 1]])
    def test_documented_location_shapes_accepted(self, loc):
        obj = {'name': 'k', 'location': loc, 'funcSrc': 'def k():\n pass\n'}
        rebuilt = PyKernelDecorator.from_json(json.dumps(obj))
        assert rebuilt.location == loc

    def test_null_location_compiles_to_clean_diagnostic(self):
        # A kernel reconstructed with a null location whose body triggers a
        # compile-time diagnostic must surface a clean CompilerError, not a raw
        # TypeError from subscripting a None location offset.
        from cudaq.kernel.ast_bridge import CompilerError
        src = 'def k():\n    nonexistent_op_xyz()\n'
        obj = {'name': 'k', 'location': None, 'funcSrc': src}
        rebuilt = PyKernelDecorator.from_json(json.dumps(obj))
        with pytest.raises(CompilerError):
            rebuilt.compile()
