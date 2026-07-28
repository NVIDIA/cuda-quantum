# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Install upstream LLVM/MLIR development artifacts into the CUDA-Q install tree
# when building the cudaq-devel wheel overlay.
if(NOT CUDAQ_BUNDLE_MLIR_INSTALL)
  return()
endif()

# LLVM_DIR points at .../lib/cmake/llvm; walk up to the install prefix.
get_filename_component(CUDAQ_LLVM_INSTALL_PREFIX "${LLVM_DIR}/../../.." ABSOLUTE)

message(STATUS "Bundling LLVM/MLIR devel tree from ${CUDAQ_LLVM_INSTALL_PREFIX}")

# Subtrees copied verbatim from the upstream LLVM/MLIR install prefix.
set(_cudaq_mlir_devel_dirs
  include
  lib
  bin
  src/python
)
foreach(_dir IN LISTS _cudaq_mlir_devel_dirs)
  if(EXISTS "${CUDAQ_LLVM_INSTALL_PREFIX}/${_dir}")
    get_filename_component(_install_destination "${_dir}" DIRECTORY)
    if(NOT _install_destination)
      set(_install_destination ".")
    endif()
    install(DIRECTORY "${CUDAQ_LLVM_INSTALL_PREFIX}/${_dir}"
            DESTINATION "${_install_destination}"
            COMPONENT Development
            USE_SOURCE_PERMISSIONS)
  endif()
endforeach()

# Make the bundled LLVM tree relocatable: clang.cfg / LLVMConfig.cmake often
# contain absolute paths from the machine that built LLVM.
# Quoted install(CODE) so ${CUDAQ_LLVM_INSTALL_PREFIX} is expanded at
# configure time into the install script.
install(CODE "
  set(_prefix \"\${CMAKE_INSTALL_PREFIX}\")
  set(_llvm_src_prefix \"${CUDAQ_LLVM_INSTALL_PREFIX}\")
  foreach(_cfg IN ITEMS clang.cfg clang++.cfg)
    set(_cfg_path \"\${_prefix}/bin/\${_cfg}\")
    if(EXISTS \"\${_cfg_path}\")
      file(READ \"\${_cfg_path}\" _cfg_content)
      # Prefer CFGDIR-relative paths so the wheel can be installed anywhere.
      string(REPLACE \"\${_llvm_src_prefix}\" \"<CFGDIR>/..\"
        _cfg_content \"\${_cfg_content}\")
      # Also rewrite any other absolute .../lib entries that may remain.
      string(REGEX REPLACE
        \"-L\\\"?/[^\\\"\\n]+/lib\\\"?\"
        \"-L\\\"<CFGDIR>/../lib\\\"\"
        _cfg_content \"\${_cfg_content}\")
      string(REGEX REPLACE
        \"-Wl,-rpath,\\\"?/[^\\\"\\n]+/lib\\\"?\"
        \"-Wl,-rpath,\\\"<CFGDIR>/../lib\\\"\"
        _cfg_content \"\${_cfg_content}\")
      file(WRITE \"\${_cfg_path}\" \"\${_cfg_content}\")
      message(STATUS \"Relocated \${_cfg} for relocatable devel install\")
    endif()
  endforeach()

  set(_llvm_config \"\${_prefix}/lib/cmake/llvm/LLVMConfig.cmake\")
  if(EXISTS \"\${_llvm_config}\")
    file(READ \"\${_llvm_config}\" _llvm_config_content)
    # Drop build-machine ZLIB_ROOT so consumers use a system/find_package zlib.
    string(REGEX REPLACE
      \"[ \\t]*set\\\\(ZLIB_ROOT [^)]*\\\\)[ \\t]*\\n?\"
      \"\"
      _llvm_config_content
      \"\${_llvm_config_content}\")
    file(WRITE \"\${_llvm_config}\" \"\${_llvm_config_content}\")
    message(STATUS \"Scrubbed ZLIB_ROOT from bundled LLVMConfig.cmake\")
  endif()
"
  COMPONENT Development)
