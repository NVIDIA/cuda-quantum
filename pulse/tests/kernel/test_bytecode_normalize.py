# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the version-isolated bytecode normalizer.

CI exercises a single interpreter, so the per-version opname maps are
checked directly here: a map that is missing an opcode the running
version emits only shows up as a ``CompilationError`` on that version.
"""

import pytest

from cudaq_pulse.kernel._bytecode_normalize import (
    _OPNAME_MAP_39,
    _OPNAME_MAP_311,
    _OPNAME_MAP_312,
    _select_map,
    normalize,
)

# Versions the bridge claims to support.
_SUPPORTED = [(3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14)]


@pytest.mark.parametrize("major,minor", _SUPPORTED)
def test_every_supported_version_has_a_map(major, minor):
    assert _select_map(major, minor)


def test_unsupported_version_raises():
    with pytest.raises(NotImplementedError):
        _select_map(3, 8)
    with pytest.raises(NotImplementedError):
        _select_map(2, 7)


@pytest.mark.parametrize("major,minor", [(3, 9), (3, 10), (3, 11)])
def test_pre_312_versions_canonicalize_load_method(major, minor):
    """LOAD_METHOD was folded into LOAD_ATTR in 3.12; earlier versions
    still emit it for ``obj.method()`` and must map it explicitly."""
    assert _select_map(major, minor)["LOAD_METHOD"] == "LOAD_ATTR"


def test_311_map_extends_the_312_map():
    for opname, canonical in _OPNAME_MAP_312.items():
        assert _OPNAME_MAP_311[opname] == canonical


def test_39_map_canonicalizes_calls():
    assert _OPNAME_MAP_39["CALL_METHOD"] == "CALL"
    assert _OPNAME_MAP_39["CALL_FUNCTION"] == "CALL"


def test_attribute_call_normalizes_to_load_attr_and_call():
    """Runs on whatever interpreter CI uses: a module-attribute call must
    canonicalize identically on every supported version."""

    class _Mod:

        @staticmethod
        def helper():
            return 1

    def attr_call():
        return _Mod.helper()

    ops = [ci.op for ci in normalize(attr_call.__code__)]
    assert "LOAD_ATTR" in ops
    assert "CALL" in ops
    assert not any(op.startswith("LOAD_METHOD") for op in ops)


def test_bare_call_normalizes_to_call():

    def bare_call():
        return len([])

    ops = [ci.op for ci in normalize(bare_call.__code__)]
    assert "CALL" in ops
    assert ops[-1] == "RETURN"
