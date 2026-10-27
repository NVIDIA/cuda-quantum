# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Check that a library is listed in `CUDAQ_MLIR_BUNDLED_LIBS`.
function(check_registered_mlir_lib name)
  if(NOT name IN_LIST CUDAQ_MLIR_BUNDLED_LIBS)
    message(FATAL_ERROR
      "${name} is not listed in ${CUDAQ_MLIR_BUNDLED_LIBS_PATH}.")
  endif()
endfunction()

add_subdirectory(Builder)
add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(CAPI)
add_subdirectory(Analysis)

# Define the shared core library libcudaqMLIR.so
include(cudaqMLIR-shlib.cmake)

# Ship the shared upstream providers with the core package: libcudaqMLIR and
# downstream consumers resolve them from core's lib directory at load time.
# Canonical SONAME files are installed as real files; link/runtime aliases
# become symlinks (wheel packaging materializes them as byte-equivalent
# copies, keeping every required name a usable ELF file).
if(CUDAQ_MLIR_SHARED_PROVIDERS)
  # Resolve the provider files at configure time: the unversioned link name
  # resolves through the upstream symlinks to the canonical (real) file.
  get_filename_component(_cudaq_llvm_prefix "${LLVM_DIR}/../../.." ABSOLUTE)
  set(_cudaq_provider_specs)
  foreach(_provider IN ITEMS LLVM MLIR MLIR-C MLIRExecutionEngineShared)
    set(_link_dir_entry
      "${_cudaq_llvm_prefix}/lib/lib${_provider}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_link_dir_entry}")
      message(FATAL_ERROR
        "Shared provider ${_link_dir_entry} not found in the toolchain.")
    endif()
    get_filename_component(_real "${_link_dir_entry}" REALPATH)
    get_filename_component(_real_name "${_real}" NAME)
    if(APPLE)
      set(_soname_name "${_real_name}")
    else()
      set(_soname_name
        "lib${_provider}${CMAKE_SHARED_LIBRARY_SUFFIX}.${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
      if(NOT EXISTS "${_cudaq_llvm_prefix}/lib/${_soname_name}")
        set(_soname_name "${_real_name}")
      endif()
    endif()
    if(_provider STREQUAL "LLVM")
      set(_cudaq_llvm_soname_name "${_soname_name}")
      set(_cudaq_llvm_compat_link_name
        "libLLVM-${LLVM_VERSION_MAJOR}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
    list(APPEND _cudaq_provider_specs
      "${_provider}|${_real}|${_real_name}|${_soname_name}|lib${_provider}${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endforeach()
  set(_cudaq_providers_script
    "${CMAKE_CURRENT_BINARY_DIR}/InstallMLIRProviders.cmake")
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/modules/InstallMLIRProviders.cmake.in"
    "${_cudaq_providers_script}" @ONLY)
  install(SCRIPT "${_cudaq_providers_script}" COMPONENT Core)
endif()

# Define a static variant of the bundled MLIR library. Only used by tools in
# the legacy static configuration (CUDAQ_MLIR_STATIC); SDK builds with shared
# providers resolve MLIR from libcudaqMLIR and the upstream providers.
add_library(cudaq-mlir-static INTERFACE)
target_link_libraries(cudaq-mlir-static INTERFACE ${CUDAQ_MLIR_BUNDLED_LIBS})
