# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Static build-policy checks for the optional shared CUDA-Q SDK path.

These tests guard source-level policy: SDK mode selects ``cudaq::MLIR``, no
static dialect fallback is present, and the typed pass is registered in the
QLX host runtime. They do not load CUDA-Q or prove a single runtime TypeID
universe. That observable contract belongs to the live-module tests in
``test_quake_import.py`` and to the shared-SDK build configuration that runs
them without skipping.
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


def _after(text: str, marker: str, *, source: str) -> str:
    _, found, suffix = text.partition(marker)
    assert found, f"{source} is missing the expected `{marker}` policy block"
    return suffix


def test_sdk_mode_enables_quake_through_the_shared_cudaq_target():
    cmake = _cmake_code("ir/lib/Conversion/QuakeToQLX/CMakeLists.txt")
    sdk_configuration = _after(cmake,
                               "if(QLX_USE_CUDAQ_SDK)",
                               source="QuakeToQLX/CMakeLists.txt")

    assert re.search(
        r"target_link_libraries\(\s*QLXCUDAQuantumDialects\s+INTERFACE\s+cudaq::MLIR\s*\)",
        sdk_configuration,
    ), "SDK mode must resolve CUDA-Q dialects through cudaq::MLIR"
    assert "NOT IS_DIRECTORY" in sdk_configuration
    assert "CUDAQ_INCLUDE_DIR" in sdk_configuration
    assert "STATIC IMPORTED" not in sdk_configuration


def test_quake_import_has_no_static_cudaq_dialect_fallback():
    cmake = _cmake_code("ir/lib/Conversion/QuakeToQLX/CMakeLists.txt")
    assert "QLX_CUDAQ_SOURCE_DIR" not in cmake
    assert "QLX_CUDAQ_BUILD_DIR" not in cmake
    assert "STATIC IMPORTED" not in cmake


def test_quake_import_constructs_an_unspecialized_executable_program():
    conversion = _source("ir/lib/Conversion/QuakeToQLX/QuakeToQLX.cpp")
    assert re.search(
        r"qlx::ProgramOp::create\(\s*builder\s*,\s*function\.getLoc\(\)\s*,"
        r"\s*\*sourceName\s*,\s*functionType\s*,\s*"
        r"/\*estimateOnly=\*/\s*nullptr\s*,\s*"
        r"/\*specialization=\*/\s*nullptr\s*\)",
        conversion,
    ), ("Quake import must pass both optional qlx.program attributes: it is "
        "executable and has no QLX workload specialization")


def test_sdk_quake_plugin_links_shared_dialects_without_force_loading_them():
    cmake = _cmake_code("ir/lib/Conversion/QuakeToQLX/Plugin/CMakeLists.txt")
    assert re.search(
        r"target_link_libraries\(\s*qlx-quake-plugin\s+PRIVATE\s+"
        r"QLXCUDAQuantumDialects\s*\)",
        cmake,
    ), "the plugin must resolve CUDA-Q dialects through the shared selector"
    assert "force_load" not in cmake
    assert "whole-archive" not in cmake
    assert "TARGET_OBJECTS:obj.MLIRQuakeToQLX" not in cmake


def test_sdk_quake_plugin_install_rewrites_the_build_tree_rpath():
    cmake = _cmake_code("python/CMakeLists.txt")
    install = _after(cmake,
                     "if(TARGET qlx-quake-plugin)",
                     source="python/CMakeLists.txt")
    wheel_rpaths = _after(cmake, "if(SKBUILD)", source="python/CMakeLists.txt")

    assert re.search(
        r"install\(\s*TARGETS\s+qlx-quake-plugin\s+LIBRARY\s+"
        r"DESTINATION\s+\"\$\{QLX_PACKAGE_INSTALL_PREFIX\}/_quake\"",
        install,
    ), "the wheel install must process the plugin as a CMake target"
    assert not re.search(
        r"install\(\s*FILES\s+\$<TARGET_FILE:qlx-quake-plugin>",
        install,
    ), "a raw file copy preserves the plugin's absolute build-tree RUNPATH"
    assert re.search(
        r"list\(\s*APPEND\s+_qlx_cudaq_runtime_targets\s+"
        r"qlx-quake-plugin\s*\)",
        wheel_rpaths,
    ), "the plugin must receive the wheel-relative CUDA-Q install RPATH"


def test_typed_pass_lives_in_the_qlx_host_runtime():
    registration = _source("ir/include/qlx/InitAllPasses.h")
    capi = _source("ir/lib/CAPI/Passes/Passes.cpp")
    cmake = _cmake_code("ir/lib/CAPI/Passes/CMakeLists.txt")
    sdk_blocks = re.findall(
        r"#ifdef\s+QLX_HAS_CUDAQ_QUAKE\s*(.*?)\s*#endif",
        registration,
        re.DOTALL,
    )
    pass_blocks = tuple(
        block for block in sdk_blocks if "registerQuakeToQLXPasses" in block)
    assert len(pass_blocks) == 1
    assert "registerPrepareQuakeForQLXPipeline" in pass_blocks[0]
    assert "qlx::registerAllQLXPasses();" in capi
    assert "list(APPEND _qlx_optional_pass_libraries MLIRQuakeToQLX)" in cmake
    assert "${_qlx_optional_pass_libraries}" in cmake
    assert not re.search(
        r"target_link_libraries\(\s*QLXCAPIPasses\s+PUBLIC\s+MLIRQuakeToQLX",
        cmake,
    ), "optional static pass libraries must be part of LINK_LIBS"


def test_quake_targets_use_the_shared_mlir_link_selector():
    conversion = _cmake_code("ir/lib/Conversion/QuakeToQLX/CMakeLists.txt")
    tool = _cmake_code("ir/tools/qlx-opt/CMakeLists.txt")
    tool_source = _source("ir/tools/qlx-opt/qlx-opt.cpp")

    assert conversion.count("qlx_select_mlir_link_libraries(") == 2
    assert "QLXCUDAQuantumDialects" in conversion
    assert "QLXCUDAQuantumDialects" in tool
    assert "${_qlx_quake_mlir_link_libraries}" in conversion
    assert "${_qlx_quake_mlir_link_libraries}" in tool
    assert "qlx::registerAllQLXPasses();" in tool_source
    assert re.search(
        r"#ifdef\s+QLX_HAS_CUDAQ_QUAKE\s+"
        r"registry.insert<cudaq::cc::CCDialect>\s*\(\s*\)\s*;\s+"
        r"registry.insert<cudaq::quake::QuakeDialect>\s*\(\s*\)\s*;\s+"
        r"#endif",
        tool_source,
    ), "qlx-opt must not register Quake dialects without SDK support"
