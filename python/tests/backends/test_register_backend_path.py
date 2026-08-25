# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""End-to-end tests for `cudaq.register_backend_path`.

These exercise the Python binding added in B1: a plugin package layout
on disk is registered with the running CUDA-Q runtime and its targets
become visible via `cudaq.get_targets()` / `cudaq.has_target()`.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cudaq


def _write_plugin(root: Path, name: str) -> Path:
    """Materialize a minimal `<root>/<name>/{targets,lib}/` plugin tree."""
    pkg = root / name
    (pkg / "targets").mkdir(parents=True)
    (pkg / "lib").mkdir(parents=True)
    (pkg / "targets" / f"{name}.yml").write_text(
        f"name: {name}\n"
        f'description: "Plugin for register_backend_path test."\n'
        f"config:\n"
        f"  platform-qpu: remote_rest\n"
        f"  library-mode: false\n")
    return pkg


def test_register_backend_path_adds_target(tmp_path):
    pkg = _write_plugin(tmp_path, "rbp-good")
    assert not cudaq.has_target("rbp-good")

    cudaq.register_backend_path(str(pkg))

    assert cudaq.has_target("rbp-good")
    names = [t.name for t in cudaq.get_targets()]
    assert "rbp-good" in names


def test_register_backend_path_rejects_missing_dir(tmp_path):
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError) as excinfo:
        cudaq.register_backend_path(str(bogus))
    assert str(bogus) in str(excinfo.value)


def test_register_backend_path_rejects_missing_targets_subdir(tmp_path):
    pkg = tmp_path / "no-targets"
    pkg.mkdir()
    with pytest.raises(RuntimeError) as excinfo:
        cudaq.register_backend_path(str(pkg))
    assert str(pkg) in str(excinfo.value)


def test_cudaq_backends_entry_point_can_register_during_import(tmp_path):
    plugin = tmp_path / "fake_cudaq_backend"
    (plugin / "targets").mkdir(parents=True)
    (plugin / "lib").mkdir()
    (plugin / "targets" / "entry-point-target.yml").write_text(
        "name: entry-point-target\n"
        'description: "Entry point registration test."\n'
        "config:\n"
        "  platform-qpu: remote_rest\n"
        "  library-mode: false\n")
    (plugin / "__init__.py").write_text(
        "from importlib.resources import files\n\n"
        "def register():\n"
        "    import cudaq\n"
        "    cudaq.register_backend_path(str(files(__name__)))\n")

    dist_info = tmp_path / "fake_cudaq_backend-0.0.dist-info"
    dist_info.mkdir()
    (dist_info /
     "METADATA").write_text("Name: fake-cudaq-backend\nVersion: 0.0\n")
    (dist_info / "entry_points.txt").write_text(
        "[cudaq.backends]\n"
        "fake = fake_cudaq_backend:register\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(tmp_path),
        str(Path.cwd() / "build" / "python"),
        env.get("PYTHONPATH", "")
    ])

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cudaq; assert cudaq.has_target('entry-point-target')",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
