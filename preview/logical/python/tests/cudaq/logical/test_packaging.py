# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Packaging contract tests: runtime extras, dynamic version, import guards."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = ROOT / "python" / "pyproject.toml"

# Mirror tests/conftest.py in subprocesses: build-tree runs extend
# cudaq.__path__ through the generated shim; wheel installs import directly.
# The shim directory is injected explicitly so subprocesses do not depend on
# PYTHONPATH, and the neutral cwd keeps the source-only copy under python/
# off sys.path (`python -c` prepends the working directory).
_IMPORT_PREAMBLE = f"""
import sys
sys.path.insert(0, {str(ROOT / "build" / "python")!r})
try:
    import _cudaq_logical_devpath  # noqa: F401
except ImportError:
    pass
"""


def _load_pyproject(path: Path = PYPROJECT) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _run_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True,
                          text=True,
                          cwd=tmp_path,
                          check=False)


def test_base_dependencies_exclude_the_cudaq_runtime():
    project = _load_pyproject()["project"]
    dependencies = project["dependencies"]
    offenders = [
        d for d in dependencies if d.startswith(("cudaq", "cuda-quantum"))
    ]
    assert offenders == []


def test_runtime_extras_map_to_the_cudaq_runtime_wheels():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert extras["cu12"] == ["cuda-quantum-cu12"]
    assert extras["cu13"] == ["cuda-quantum-cu13"]


def test_version_is_dynamic_and_uses_logical_tags():
    from packaging.version import Version

    import cudaq.logical

    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert "version" not in project
    assert "version" in project["dynamic"]

    provider = pyproject["tool"]["scikit-build"]["metadata"]["version"]
    assert provider["provider"].endswith(".metadata.setuptools_scm")

    scm = pyproject["tool"]["setuptools_scm"]
    assert "cudaq-logical/v" in scm["tag_regex"]
    assert scm["git_describe_command"][-1] == "cudaq-logical/v*"
    assert str(Version(cudaq.logical.__version__))


def test_preview_warning_is_emitted_once_on_import(tmp_path):
    code = _IMPORT_PREAMBLE + """
import cudaq.logical
import cudaq.logical  # cached in sys.modules: must not re-warn
"""
    result = _run_python(tmp_path, code)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("cudaq-logical is in preview") == 1


def test_missing_cudaq_runtime_raises_helpful_import_error(tmp_path):
    code = """
import importlib.util
_real_find_spec = importlib.util.find_spec

def _without_cudaq(name, *args, **kwargs):
    if name == "cudaq" or name.startswith("cudaq."):
        return None
    return _real_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _without_cudaq
""" + _IMPORT_PREAMBLE + "import cudaq.logical\n"
    result = _run_python(tmp_path, code)
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    for hint in ("cudaq-logical[cu13]", "cudaq-logical[cu12]",
                 "pip install cudaq"):
        assert hint in result.stderr


def test_stamp_script_pins_both_runtime_extras(tmp_path):
    from scripts.stamp_cudaq_runtime_dependency import stamp_runtime_dependency

    stamped = tmp_path / "pyproject.toml"
    stamped.write_text(PYPROJECT.read_text())
    for distribution in ("cuda-quantum-cu12", "cuda-quantum-cu13"):
        dependency = stamp_runtime_dependency(stamped,
                                              distribution=distribution,
                                              version="0.16.0")
        assert dependency == f"{distribution}==0.16.0"

    project = _load_pyproject(stamped)["project"]
    extras = project["optional-dependencies"]
    assert extras["cu12"] == ["cuda-quantum-cu12==0.16.0"]
    assert extras["cu13"] == ["cuda-quantum-cu13==0.16.0"]
    # The base dependency list stays runtime-free after stamping.
    assert not any(
        d.startswith(("cudaq", "cuda-quantum"))
        for d in project["dependencies"])


def test_stamp_script_requires_exactly_one_bare_entry_per_distribution(
        tmp_path):
    from scripts.stamp_cudaq_runtime_dependency import stamp_runtime_dependency

    missing = tmp_path / "missing.toml"
    missing.write_text('[project]\ndependencies = [\n  "stim",\n]\n')
    with pytest.raises(RuntimeError, match="exactly one bare"):
        stamp_runtime_dependency(missing,
                                 distribution="cuda-quantum-cu12",
                                 version="0.16.0")

    duplicated = tmp_path / "duplicated.toml"
    duplicated.write_text("[project.optional-dependencies]\n"
                          'cu12 = [\n  "cuda-quantum-cu12",\n]\n'
                          'cu12-alias = [\n  "cuda-quantum-cu12",\n]\n')
    with pytest.raises(RuntimeError, match="found 2"):
        stamp_runtime_dependency(duplicated,
                                 distribution="cuda-quantum-cu12",
                                 version="0.16.0")
