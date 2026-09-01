# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Static build-policy checks for the shared CUDA-Q compiler path.

These tests guard source-level policy: CUDA-Q Logical resolves MLIR through
``cudaq::MLIR``, no static dialect fallback is present, and the typed pass is
registered in the CUDA-Q Logical host runtime. They do not load CUDA-Q or prove a single
runtime TypeID universe. That observable contract belongs to the live-module
tests in ``test_quake_import.py``.
"""

import re
from pathlib import Path

repo = Path(__file__).resolve().parents[4]


def _source(relative: str) -> str:
    return (repo / relative).read_text()


def _cmake_code(relative: str) -> str:
    """Return comment-free CMake with insignificant whitespace normalized."""
    uncommented = "\n".join(
        line.split("#", 1)[0] for line in _source(relative).splitlines())
    return re.sub(r"\s+", " ", uncommented).strip()


def test_quake_is_enabled_through_the_shared_cudaq_target():
    cmake = _cmake_code("ir/lib/Conversion/QuakeToQLX/CMakeLists.txt")

    assert re.search(
        r"target_link_libraries\(\s*QLXCUDAQuantumDialects\s+INTERFACE\s+cudaq::MLIR\s*\)",
        cmake,
    ), "CUDA-Q dialects must resolve through cudaq::MLIR"
    assert "NOT IS_DIRECTORY" in cmake
    assert "CUDAQ_INCLUDE_DIR" in cmake
    assert "STATIC IMPORTED" not in cmake


def test_quake_import_has_no_static_cudaq_dialect_fallback():
    cmake = _cmake_code("ir/lib/Conversion/QuakeToQLX/CMakeLists.txt")
    assert "QLX_CUDAQ_SOURCE_DIR" not in cmake
    assert "QLX_CUDAQ_BUILD_DIR" not in cmake
    assert "STATIC IMPORTED" not in cmake


def test_quake_plugin_links_shared_dialects_without_force_loading_them():
    cmake = _cmake_code("ir/lib/Conversion/QuakeToQLX/Plugin/CMakeLists.txt")
    assert re.search(
        r"target_link_libraries\(\s*qlx-quake-plugin\s+PRIVATE\s+"
        r"QLXCUDAQuantumDialects\s*\)",
        cmake,
    ), "the plugin must resolve CUDA-Q dialects through the shared selector"
    assert "force_load" not in cmake
    assert "whole-archive" not in cmake
    assert "TARGET_OBJECTS:obj.MLIRQuakeToQLX" not in cmake


def test_typed_pass_lives_in_the_qlx_host_runtime():
    registration = _source("ir/include/qlx/InitAllPasses.h")
    capi = _source("ir/lib/CAPI/Passes/Passes.cpp")
    cmake = _cmake_code("ir/lib/CAPI/Passes/CMakeLists.txt")

    assert "registerQuakeToQLXPasses();" in registration
    assert "registerPrepareQuakeForQLXPipeline();" in registration
    assert "qlx::registerAllQLXPasses();" in capi
    assert "MLIRQuakeToQLX" in cmake
    assert not re.search(
        r"target_link_libraries\(\s*QLXCAPIPasses\s+PUBLIC\s+MLIRQuakeToQLX",
        cmake,
    ), "the static pass library must be part of LINK_LIBS"


def test_no_build_time_switch_selects_the_mlir_link_surface():
    """CUDA-Q Logical has exactly one MLIR link surface, chosen by CUDA-Q rather than CUDA-Q Logical.

    Libraries name the MLIR components they use, like any other MLIR project;
    bundled components resolve from `libcudaqMLIR` through `cudaq::cudaqMLIR`'s
    interface link options. Nothing here selects between a static and a shared
    MLIR link surface.
    """
    for relative in (
            "ir/lib/Conversion/QuakeToQLX/CMakeLists.txt",
            "ir/lib/Dialect/QLX/IR/CMakeLists.txt",
            "ir/tools/qlx-opt/CMakeLists.txt",
            "python/CMakeLists.txt",
    ):
        cmake = _cmake_code(relative)
        assert "QLX_USE_CUDAQ_SDK" not in cmake, relative
        assert "QLX_BUILD_BUNDLED_LLVM" not in cmake, relative
        assert "qlx_select_mlir_link_libraries" not in cmake, relative

    # Representative libraries declare real components rather than cudaq::MLIR.
    dialect = _cmake_code("ir/lib/Dialect/QLX/IR/CMakeLists.txt")
    assert "MLIRIR" in dialect and "MLIRSupport" in dialect
    assert "cudaq::MLIR" not in dialect


def test_qlx_opt_registers_the_cudaq_dialects():
    tool = _cmake_code("ir/tools/qlx-opt/CMakeLists.txt")
    tool_source = _source("ir/tools/qlx-opt/qlx-opt.cpp")

    assert "QLXCUDAQuantumDialects" in tool
    assert "qlx::registerAllQLXPasses();" in tool_source
    assert "registry.insert<cudaq::cc::CCDialect>();" in tool_source
    assert "registry.insert<cudaq::quake::QuakeDialect>();" in tool_source
