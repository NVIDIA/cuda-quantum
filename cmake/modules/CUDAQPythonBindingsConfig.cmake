# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Marker for whether the CUDAQ install this find_package(CUDAQ) resolved was
# built with -DCUDAQ_ENABLE_PYTHON_BINDINGS=ON (see cudaq/python/CMakeLists.txt
# and the top-level CMakeLists.txt). CUDAQPythonBindingsTargets.cmake is only
# installed by that build path, so its presence is the signal: a python/
# build configured independently of cudaq/ (i.e. against an installed CUDAQ,
# not built together with cudaq/ in the same configure) checks
# CUDAQ_ENABLE_PYTHON_BINDINGS after find_package(CUDAQ) to fail with a clear
# diagnostic rather than a confusing missing-target error.

get_filename_component(CUDAQ_PYTHONBINDINGS_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(EXISTS "${CUDAQ_PYTHONBINDINGS_CMAKE_DIR}/CUDAQPythonBindingsTargets.cmake")
  set(CUDAQ_ENABLE_PYTHON_BINDINGS TRUE)
  if(NOT TARGET cudaq::cudaqMLIRCAPI)
    include("${CUDAQ_PYTHONBINDINGS_CMAKE_DIR}/CUDAQPythonBindingsTargets.cmake")
  endif()
else()
  set(CUDAQ_ENABLE_PYTHON_BINDINGS FALSE)
endif()
