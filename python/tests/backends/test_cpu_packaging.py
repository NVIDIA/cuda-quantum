# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for CPU-only packaging, autodetection, and backward compatibility."""

import importlib.util
import os
from pathlib import Path
import sys
import tomllib
import types
import unittest.mock as mock
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_hatch_build():
    """Load hatch_build.py directly from python/metapackages/, mocking hatchling if absent."""
    if "hatchling.metadata.plugin.interface" not in sys.modules:
        hatchling_mod = types.ModuleType("hatchling")
        metadata_mod = types.ModuleType("hatchling.metadata")
        plugin_mod = types.ModuleType("hatchling.metadata.plugin")
        interface_mod = types.ModuleType("hatchling.metadata.plugin.interface")

        class MetadataHookInterface:
            def __init__(self, root=None, config=None):
                self.root = root
                self.config = config

        interface_mod.MetadataHookInterface = MetadataHookInterface
        sys.modules["hatchling"] = hatchling_mod
        sys.modules["hatchling.metadata"] = metadata_mod
        sys.modules["hatchling.metadata.plugin"] = plugin_mod
        sys.modules["hatchling.metadata.plugin.interface"] = interface_mod

    hatch_build_path = REPO_ROOT / "python" / "metapackages" / "hatch_build.py"
    spec = importlib.util.spec_from_file_location("hatch_build", hatch_build_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_pyproject_metadata():
    """Verify pyproject.toml.cpu contains no mandatory CUDA/cuQuantum dependencies."""
    cpu_toml = REPO_ROOT / "pyproject.toml.cpu"
    assert cpu_toml.exists(), "pyproject.toml.cpu must exist"

    with cpu_toml.open("rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    assert project["name"] == "cuda-quantum-cpu"

    deps = project.get("dependencies", [])
    # Verify core required CPU dependencies
    assert any("numpy" in d for d in deps), "numpy must be a dependency"
    assert any("scipy" in d for d in deps), "scipy must be a dependency"
    assert any("astpretty" in d for d in deps), "astpretty must be a dependency"
    assert any("requests" in d for d in deps), "requests must be a dependency"

    # Verify NO CUDA or cuQuantum dependencies
    gpu_packages = [
        "custatevec",
        "cutensornet",
        "cudensitymat",
        "nvidia-cublas",
        "nvidia-curand",
        "nvidia-cusparse",
        "nvidia-cuda-runtime",
        "nvidia-cusolver",
        "nvidia-cuda-nvrtc",
        "cupy",
    ]
    for dep in deps:
        for gpu_pkg in gpu_packages:
            assert gpu_pkg not in dep.lower(), (
                f"CPU package must not contain GPU dependency: {dep}"
            )

    # Verify components match full frontend
    scikit_build = data["tool"]["scikit-build"]
    components = scikit_build["install"]["components"]
    assert "CUDAQCorePythonModules" in components
    assert "CUDAQuantumPythonModules" in components
    assert "Runtime" in components

    # Verify CUDA Toolkit discovery is explicitly disabled for CPU builds
    cmake_args = scikit_build.get("cmake", {}).get("args", [])
    assert "-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE" in cmake_args, (
        "pyproject.toml.cpu must disable CUDAToolkit to prevent accidental GPU backend compilation"
    )


def test_cuda_wheels_preserve_dependencies():
    """Verify cuda-quantum-cu12 and cu13 retain all their GPU dependencies."""
    cu12_toml = REPO_ROOT / "pyproject.toml.cu12"
    cu13_toml = REPO_ROOT / "pyproject.toml.cu13"
    assert cu12_toml.exists()
    assert cu13_toml.exists()

    with cu12_toml.open("rb") as f:
        cu12_data = tomllib.load(f)
    with cu13_toml.open("rb") as f:
        cu13_data = tomllib.load(f)

    cu12_deps = [d.lower() for d in cu12_data["project"]["dependencies"]]
    cu13_deps = [d.lower() for d in cu13_data["project"]["dependencies"]]

    # Verify cu12 retains its GPU dependencies
    assert any("custatevec-cu12" in d for d in cu12_deps)
    assert any("cutensornet-cu12" in d for d in cu12_deps)
    assert any("cudensitymat-cu12" in d for d in cu12_deps)
    assert any("nvidia-cublas-cu12" in d for d in cu12_deps)
    assert any("cupy-cuda12x" in d for d in cu12_deps)

    # Verify cu13 retains its GPU dependencies
    assert any("custatevec-cu13" in d for d in cu13_deps)
    assert any("cutensornet-cu13" in d for d in cu13_deps)
    assert any("cudensitymat-cu13" in d for d in cu13_deps)
    assert any("nvidia-cublas" in d for d in cu13_deps)
    assert any("cupy-cuda13x" in d for d in cu13_deps)


def test_metapackage_infer_best_package_no_cuda():
    """On Linux without CUDA, infer_best_package must return cuda-quantum-cpu."""
    hb = _load_hatch_build()

    with mock.patch.object(hb, "_get_cuda_version", return_value=None), \
         mock.patch.object(hb, "_check_package_installed", return_value=False), \
         mock.patch.object(sys, "platform", "linux"), \
         mock.patch.dict(os.environ, {}, clear=True):
        selected = hb._infer_best_package()
        assert selected == "cuda-quantum-cpu"


def test_metapackage_infer_best_package_macos_fallback():
    """On macOS without CUDA, infer_best_package preserves cu13 default."""
    hb = _load_hatch_build()

    with mock.patch.object(hb, "_get_cuda_version", return_value=None), \
         mock.patch.object(hb, "_check_package_installed", return_value=False), \
         mock.patch.object(sys, "platform", "darwin"), \
         mock.patch.dict(os.environ, {}, clear=True):
        selected = hb._infer_best_package()
        assert selected == "cuda-quantum-cu13"

    # But if cuda-quantum-cpu is already installed on macOS, it respects it:
    def check_pkg(name):
        return name == "cuda-quantum-cpu"

    with mock.patch.object(hb, "_get_cuda_version", return_value=None), \
         mock.patch.object(hb, "_check_package_installed", side_effect=check_pkg), \
         mock.patch.object(sys, "platform", "darwin"), \
         mock.patch.dict(os.environ, {}, clear=True):
        selected = hb._infer_best_package()
        assert selected == "cuda-quantum-cpu"


def test_metapackage_infer_best_package_cuda12():
    """When CUDA 12 is detected, infer_best_package must select cuda-quantum-cu12."""
    hb = _load_hatch_build()

    with mock.patch.object(hb, "_get_cuda_version", return_value=12060), \
         mock.patch.object(hb, "_check_package_installed", return_value=False), \
         mock.patch.object(sys, "platform", "linux"), \
         mock.patch.dict(os.environ, {}, clear=True):
        selected = hb._infer_best_package()
        assert selected == "cuda-quantum-cu12"


def test_metapackage_infer_best_package_cuda13():
    """When CUDA 13 is detected, infer_best_package must select cuda-quantum-cu13."""
    hb = _load_hatch_build()

    with mock.patch.object(hb, "_get_cuda_version", return_value=13000), \
         mock.patch.object(hb, "_check_package_installed", return_value=False), \
         mock.patch.object(sys, "platform", "linux"), \
         mock.patch.dict(os.environ, {}, clear=True):
        selected = hb._infer_best_package()
        assert selected == "cuda-quantum-cu13"


def test_metapackage_infer_preserves_installed_cuda_variant():
    """If a user already has cuda-quantum-cu12 installed, keep it even without CUDA."""
    hb = _load_hatch_build()

    def check_pkg(name):
        return name == "cuda-quantum-cu12"

    with mock.patch.object(hb, "_get_cuda_version", return_value=None), \
         mock.patch.object(hb, "_check_package_installed", side_effect=check_pkg), \
         mock.patch.object(sys, "platform", "linux"), \
         mock.patch.dict(os.environ, {}, clear=True):
        selected = hb._infer_best_package()
        assert selected == "cuda-quantum-cu12"


def test_metapackage_env_override():
    """CUDAQ_BDIST environment variable overrides package inference."""
    hb = _load_hatch_build()

    with mock.patch.dict(os.environ, {"CUDAQ_BDIST": "cuda-quantum-cpu"}):
        assert hb._infer_best_package() == "cuda-quantum-cpu"

    with mock.patch.dict(os.environ, {"CUDAQ_BDIST": "cuda-quantum-cu12"}):
        assert hb._infer_best_package() == "cuda-quantum-cu12"


def test_metapackage_metadata_hook_cpu():
    """Verify CudaqMetadataHook sets correct dependencies for cuda-quantum-cpu."""
    hb = _load_hatch_build()
    hook = hb.CudaqMetadataHook(None, None)

    v_file = REPO_ROOT / "python" / "metapackages" / "_version.txt"
    lv_file = REPO_ROOT / "python" / "metapackages" / "_logical_version.txt"
    try:
        v_file.write_text("0.11.0\n")
        lv_file.write_text("0.2.0\n")
        metadata = {}
        with mock.patch.object(hb, "_infer_best_package", return_value="cuda-quantum-cpu"), \
             mock.patch.dict(os.environ, {"CUDAQ_META_SDIST_BUILD": "0"}):
            hook.update(metadata)
        deps = metadata.get("dependencies", [])
        assert "cuda-quantum-cpu==0.11.0" in deps
        assert "cudaq-logical==0.2.0; sys_platform == 'linux'" in deps
    finally:
        v_file.unlink(missing_ok=True)
        lv_file.unlink(missing_ok=True)


def test_metapackage_metadata_hook_fallback_safe():
    """Verify CudaqMetadataHook safely handles unexpected bdist without KeyError."""
    hb = _load_hatch_build()
    hook = hb.CudaqMetadataHook(None, None)

    v_file = REPO_ROOT / "python" / "metapackages" / "_version.txt"
    lv_file = REPO_ROOT / "python" / "metapackages" / "_logical_version.txt"
    try:
        v_file.write_text("0.11.0\n")
        lv_file.write_text("0.2.0\n")
        metadata = {}
        with mock.patch.object(hb, "_infer_best_package", return_value="cuda-quantum-unknown"), \
             mock.patch.dict(os.environ, {"CUDAQ_META_SDIST_BUILD": "0"}):
            # Should not raise KeyError
            hook.update(metadata)
        deps = metadata.get("dependencies", [])
        assert "cuda-quantum-unknown==0.11.0" in deps
        assert "cudaq-logical[cu13]==0.2.0; sys_platform == 'linux'" in deps
    finally:
        v_file.unlink(missing_ok=True)
        lv_file.unlink(missing_ok=True)


def test_build_wheel_auditwheel_and_variant_handling():
    """Verify scripts/build_wheel.sh does not exclude CUDA libraries for CPU wheel."""
    script_path = REPO_ROOT / "scripts" / "build_wheel.sh"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")

    # Verify -c variant handling includes cpu, 12, cu12, 13, cu13
    assert '12|cu12)' in content
    assert '13|cu13)' in content
    assert 'cpu)' in content

    # Verify auditwheel exclusion of CUDA libraries is guarded to not apply to CPU
    assert 'if [ "$cuda_variant" != "cpu" ]; then' in content
    assert 'auditwheel_args="$auditwheel_args --exclude libcustatevec.so.1"' in content


def test_cpu_kernel_simulation():
    """Test minimal kernel execution and CPU target if compiled cudaq runtime is available."""
    try:
        import cudaq
        if not hasattr(cudaq, "set_target"):
            pytest.skip("cudaq runtime C++ extensions not compiled in this environment")
    except ImportError:
        pytest.skip("cudaq is not installed in the test execution environment")

    cudaq.set_target("qpp-cpu")
    assert cudaq.get_target().name == "qpp-cpu"

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        mz(q)

    counts = cudaq.sample(bell_pair, shots_count=100)
    assert "00" in counts or "11" in counts
