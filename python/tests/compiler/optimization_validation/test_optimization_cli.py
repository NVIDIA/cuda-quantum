# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the CLI's kernel-file --input handling."""

import contextlib
import json

import pytest

from cudaq._compiler import optimization_cli as cli

_KERNEL_SOURCE = '''
import cudaq


@cudaq.kernel
def alpha():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@cudaq.kernel
def beta():
    q = cudaq.qvector(1)
    for _ in range(2):
        t(q[0])
'''


@pytest.fixture
def kernel_file(tmp_path):
    path = tmp_path / "user_kernels.py"
    path.write_text(_KERNEL_SOURCE)
    return path


# A .py path or a ::name selector is a kernel spec; a bare .qke is not.
def test_is_kernel_spec():
    assert cli._is_kernel_spec("foo.py")
    assert cli._is_kernel_spec("foo.py::alpha")
    assert cli._is_kernel_spec("foo.qke::alpha")  # explicit selector wins
    assert not cli._is_kernel_spec("foo.qke")
    assert not cli._is_kernel_spec("dir/circuit.mlir")


def test_lower_kernel_spec_all(kernel_file):
    lowered = cli._lower_kernel_spec(str(kernel_file))
    names = [n for n, _ in lowered]
    assert names == ["alpha", "beta"]
    for _, text in lowered:
        assert "quake.mangled_name_map" in text  # real frontend output


def test_lower_kernel_spec_selects_one(kernel_file):
    lowered = cli._lower_kernel_spec(f"{kernel_file}::beta")
    assert [n for n, _ in lowered] == ["beta"]


def test_lower_kernel_spec_unknown_name(kernel_file):
    with pytest.raises(cli.InputError):
        cli._lower_kernel_spec(f"{kernel_file}::missing")


def test_lower_kernel_spec_missing_file(tmp_path):
    with pytest.raises(cli.InputError):
        cli._lower_kernel_spec(str(tmp_path / "nope.py"))


def test_resolve_inputs_materializes_named_qke(kernel_file):
    with contextlib.ExitStack() as stack:
        paths = cli._resolve_inputs([str(kernel_file)], stack)
        assert [p.name for p in paths] == ["alpha.qke", "beta.qke"]
        assert all(p.read_text().strip() for p in paths)
    # The temp directory is torn down when the stack closes.
    assert not any(p.exists() for p in paths)


def test_resolve_inputs_passes_qke_through():
    with contextlib.ExitStack() as stack:
        paths = cli._resolve_inputs(["some/circuit.qke"], stack)
        assert [str(p) for p in paths] == ["some/circuit.qke"]


# End-to-end main(): a kernel file validates to a passing JSON result, exit 0.
def test_main_validates_kernel_file(kernel_file, tmp_path, capsys):
    result_path = tmp_path / "result.json"
    rc = cli.main([
        "--input",
        str(kernel_file),
        "--prepare",
        "builtin.module(func.func(cc-loop-unroll,canonicalize,memtoreg))",
        "--candidate",
        "builtin.module(func.func(canonicalize))",
        "--result",
        str(result_path),
    ])
    assert rc == 0
    payload = json.loads(result_path.read_text())
    assert payload["status"] == "passed"
    inputs = sorted(c["input"].rsplit("/", 1)[-1] for c in payload["cases"])
    assert inputs == ["alpha.qke", "beta.qke"]


# A bad kernel spec is an invalid request (exit 3), reported as JSON.
def test_main_unknown_kernel_is_invalid_request(kernel_file, tmp_path):
    result_path = tmp_path / "result.json"
    rc = cli.main([
        "--input",
        f"{kernel_file}::missing",
        "--candidate",
        "builtin.module(func.func(canonicalize))",
        "--result",
        str(result_path),
    ])
    assert rc == 3
    payload = json.loads(result_path.read_text())
    assert payload["status"] == "invalid-request"
