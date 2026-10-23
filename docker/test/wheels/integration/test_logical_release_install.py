# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Exercise the release installer with real offline pip resolution."""

import base64
import csv
from email.parser import Parser
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[4]
INSTALLER = ROOT / "docker/release/install_logical_wheel.py"


def _wheel(directory, name, version, files, requires=(), tag="py3-none-any"):
    metadata = f"{name}-{version}.dist-info"
    files = dict(files)
    files[f"{metadata}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\n"
        f"Version: {version}\n" +
        "".join(f"Requires-Dist: {requirement}\n" for requirement in requires))
    files[f"{metadata}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: release-installer-test\n"
        f"Root-Is-Purelib: true\nTag: {tag}\n")
    records = io.StringIO()
    writer = csv.writer(records)
    for filename, contents in files.items():
        contents = contents.encode()
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(contents).digest()).rstrip(b"=").decode()
        writer.writerow((filename, f"sha256={digest}", len(contents)))
    writer.writerow((f"{metadata}/RECORD", "", ""))
    files[f"{metadata}/RECORD"] = records.getvalue()
    path = directory / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for filename, contents in files.items():
            wheel.writestr(filename, contents)
    return path


def _edem(directory, version="0.1.0", marker="candidate", **kwargs):
    return _wheel(
        directory, "cudaq_edem", version, {
            "cudaq_edem/__init__.py":
                f"__version__ = {version!r}\nmarker = {marker!r}\n",
            "cudaq_edem/lib/native-placeholder":
                marker,
        }, **kwargs)


@pytest.fixture(scope="module")
def python(tmp_path_factory):
    environment = tmp_path_factory.mktemp("installer-python") / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    return environment / "bin/python"


@pytest.fixture
def package(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    prefix = tmp_path / "native"
    (prefix / "cudaq").mkdir(parents=True)
    (prefix / "cudaq/__init__.py").write_text("# Existing native CUDA-Q.\n")

    def logical(requirement="cudaq-edem>=0.1,<0.2", extra=()):
        return _wheel(
            wheels, "cudaq_logical", "0.1.0", {
                "cudaq/logical/__init__.py": "# Logical.\n",
                "cudaq/mlir/dialects/fabric.py": "# Fabric.\n",
                "cudaq/mlir/_mlir_libs/native-placeholder": "native",
                "bin/qlx-opt": "# Tool.\n",
            }, ("cudaq-core==0.0.0", requirement, *extra))

    return wheels, prefix, logical


def _install(python, wheels, prefix, **pip_environment):
    environment = dict(os.environ, PIP_NO_INDEX="1", PIP_FIND_LINKS=str(wheels))
    environment.update(pip_environment)
    return subprocess.run([
        str(python),
        str(INSTALLER), "--wheel-dir",
        str(wheels), "--prefix",
        str(prefix)
    ],
                          capture_output=True,
                          text=True,
                          env=environment)


def test_installs_matching_native_dependency_beside_logical(python, package):
    wheels, prefix, logical = package
    logical()
    _edem(wheels)
    _edem(wheels, "0.2.0", "incompatible")
    result = _install(python, wheels, prefix)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (prefix /
            "cudaq_edem/lib/native-placeholder").read_text() == "candidate"
    assert not (prefix / "cudaq_edem-0.2.0.dist-info").exists()
    assert (prefix /
            "cudaq/__init__.py").read_text() == "# Existing native CUDA-Q.\n"
    metadata = Parser().parsestr(
        (prefix / "cudaq_logical-0.1.0.dist-info/METADATA").read_text())
    assert metadata.get_all("Requires-Dist") == ["cudaq-edem>=0.1,<0.2"]
    owners = {}
    for name in ("cudaq_edem", "cudaq_logical"):
        records = prefix / f"{name}-0.1.0.dist-info/RECORD"
        with records.open() as handle:
            owners[name] = {row[0] for row in csv.reader(handle)}
    assert "cudaq_edem/lib/native-placeholder" in owners["cudaq_edem"]
    assert "cudaq_edem/lib/native-placeholder" not in owners["cudaq_logical"]
    assert "cudaq/__init__.py" not in owners["cudaq_logical"]
    assert not owners["cudaq_edem"] & owners["cudaq_logical"]


@pytest.mark.parametrize("version, tag",
                         [(None, "py3-none-any"), ("0.2.0", "py3-none-any"),
                          ("0.1.0", "cp399-cp399-linux_x86_64")])
def test_rejects_missing_incompatible_or_wrong_python_edem(
        python, package, version, tag):
    wheels, prefix, logical = package
    logical()
    if version is not None:
        _edem(wheels, version, tag=tag)
    result = _install(python, wheels, prefix)
    assert result.returncode != 0
    assert "No matching distribution found for cudaq-edem" in result.stderr
    assert not (prefix / "cudaq/logical").exists()
    assert not (prefix / "cudaq_edem").exists()


def test_local_candidate_wins_over_same_version_installed_package(
        python, package, tmp_path):
    wheels, prefix, logical = package
    logical()
    _edem(wheels)
    stale = tmp_path / "stale"
    stale.mkdir()
    previous = _edem(stale, marker="stale")
    subprocess.run([
        str(python), "-m", "pip", "install", "--no-index", "--no-deps",
        str(previous)
    ],
                   check=True,
                   capture_output=True,
                   text=True)
    result = _install(python, wheels, prefix)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (prefix /
            "cudaq_edem/lib/native-placeholder").read_text() == "candidate"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "import cudaq_edem; print(cudaq_edem.marker)")
    selected = subprocess.check_output(
        [str(python), "-I", "-c", code,
         str(prefix)], text=True)
    assert selected.strip() == "candidate"


def test_environment_wheelhouse_cannot_replace_candidate(
        python, package, tmp_path):
    wheels, prefix, logical = package
    logical()
    _edem(wheels)
    unrelated = tmp_path / "other-build"
    unrelated.mkdir()
    _edem(unrelated, "0.1.1", "other-build")
    result = _install(python, wheels, prefix, PIP_FIND_LINKS=str(unrelated))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (prefix /
            "cudaq_edem/lib/native-placeholder").read_text() == "candidate"
    assert not (prefix / "cudaq_edem-0.1.1.dist-info").exists()


def test_other_dependencies_keep_normal_pip_resolution(python, package):
    wheels, prefix, logical = package
    logical(extra=("cudaq-core-tools==1.0",))
    _edem(wheels)
    _wheel(wheels, "cudaq_core_tools", "1.0", {"core_tools.py": "value = 7\n"})
    result = _install(python, wheels, prefix)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = Parser().parsestr(
        (prefix / "cudaq_logical-0.1.0.dist-info/METADATA").read_text())
    assert metadata.get_all("Requires-Dist") == [
        "cudaq-edem>=0.1,<0.2", "cudaq-core-tools==1.0"
    ]
    subprocess.run(
        [str(python), "-c", "import core_tools; assert core_tools.value == 7"],
        check=True)
