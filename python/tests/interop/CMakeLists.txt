# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()

add_compile_options(-Wno-error)
set(CMAKE_INSTALL_RPATH
  "${CMAKE_INSTALL_RPATH}:${_origin_prefix}:${CMAKE_BINARY_DIR}/lib")

add_subdirectory(quantum_lib)

pybind11_add_module(cudaq_test_cpp_algo
  test_cpp_quantum_algorithm_module.cpp
)

target_link_libraries(cudaq_test_cpp_algo
  PRIVATE
    cudaq
    quantum_lib
    cudaq-python-interop
)

target_include_directories(cudaq_test_cpp_algo
  PRIVATE
    ${CMAKE_SOURCE_DIR}/python
)
  
add_dependencies(cudaq_test_cpp_algo nvq++)

