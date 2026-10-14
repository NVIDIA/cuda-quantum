# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Quake-import plugin: the CUDA-Q Quake/CC dialects packaged as a loadable
# MLIR DialectPlugin. A no-op PassPlugin entry point keeps the generic Logical
# plugin loader compatible; the typed conversion pass lives in _qlxRuntime so
# it shares the host Logical operation TypeIDs. Loaded at runtime by
# _qlxRuntime.load_plugin. The plugin links cudaq::MLIR, so all of its MLIR and
# Quake/CC symbols resolve from libcudaqMLIR via DT_NEEDED -- no RTLD_GLOBAL
# promotion of host images is needed or performed. The plugin consumes Quake/CC
# from CUDA-Q's shared MLIR target.
#
# This is a _qlxRuntime plugin only -- NOT a qlx-opt --load-*-plugin target
# (qlx-opt already links the pass + Quake/CC itself).

add_library(qlx-quake-plugin SHARED
  QuakeImportPlugin.cpp
)

# Do not copy CUDA-Q dialect objects into this plugin. Resolve them through
# cudaq::MLIR so CUDA-Q and QLX retain one Quake/CC TypeID universe.
target_link_libraries(qlx-quake-plugin PRIVATE
  QLXCUDAQuantumDialects
)
if(APPLE)
  # Defensive: should any symbol not be bound through the linked libraries
  # (two-level namespace), fall back to a flat lookup at dlopen time.
  target_link_options(qlx-quake-plugin PRIVATE
    "LINKER:-undefined,dynamic_lookup")
endif()


target_compile_options(qlx-quake-plugin PRIVATE
  -fvisibility=hidden
  -fvisibility-inlines-hidden
)

qlx_propagate_llvm_rtti_eh(qlx-quake-plugin)

set_target_properties(qlx-quake-plugin PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Copy the plugin into the build-tree Python package (the one actually imported:
# ${PROJECT_BINARY_DIR}/python/cudaq/logical) so import_quake finds
# it without a `cmake --install` step.
set(_quake_py_pkg "${PROJECT_BINARY_DIR}/python/cudaq/logical/_quake")
add_custom_command(TARGET qlx-quake-plugin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_quake_py_pkg}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:qlx-quake-plugin>
    "${_quake_py_pkg}/$<TARGET_FILE_NAME:qlx-quake-plugin>"
  COMMENT "Copying qlx-quake-plugin into python/cudaq/logical/_quake/"
)
