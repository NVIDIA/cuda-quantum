# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file sets the basic flags for the CUDA Quantum compiler
set (CUDAQ_COMPILER_EXTRA_FLAGS "")
if (CUDAQ_VERBOSE)
  set (CUDAQ_COMPILER_EXTRA_FLAGS "${CUDAQ_COMPILER_EXTRA_FLAGS} -v")
endif()

set (CUDAQ_LINKER_EXTRA_FLAGS "")
if (CUDAQ_VERBOSE)
  set (CUDAQ_LINKER_EXTRA_FLAGS "${CUDAQ_LINKER_EXTRA_FLAGS} -v")
endif()

if(NOT CMAKE_CUDAQ_COMPILE_OBJECT)
    set(CMAKE_CUDAQ_COMPILE_OBJECT "<CMAKE_CUDAQ_COMPILER> --cmake-host-compiler <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> -c <SOURCE> -o <OBJECT> ${CUDAQ_COMPILER_EXTRA_FLAGS}")
endif()
if(NOT CMAKE_CUDAQ_LINK_EXECUTABLE)
    set(CMAKE_CUDAQ_LINK_EXECUTABLE "<CMAKE_CUDAQ_COMPILER> <CMAKE_CXX_LINK_FLAGS> <LINK_LIBRARIES> <LINK_FLAGS> <OBJECTS> -o <TARGET> ${CUDAQ_LINKER_EXTRA_FLAGS}")
endif()

if(NOT CMAKE_INCLUDE_FLAG_CUDAQ)
  set (CMAKE_INCLUDE_FLAG_CUDAQ "-I")
endif()
# TODO: add CMAKE_CUDAQ_CREATE_SHARED_LIBRARY and CMAKE_CUDAQ_CREATE_STATIC_LIBRARY rules.

set(CMAKE_CUDAQ_INFORMATION_LOADED 1)
