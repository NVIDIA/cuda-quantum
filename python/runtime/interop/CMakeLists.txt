# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_compile_options(-Wno-attributes)
nanobind_build_library(nanobind-static)
add_library(cudaq-python-interop SHARED PythonCppInterop.cpp)
target_include_directories(cudaq-python-interop PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${nanobind_INCLUDE_DIR}
)
target_link_libraries(cudaq-python-interop PRIVATE nanobind-static Python3::Module cudaq)
install (FILES PythonCppInterop.h DESTINATION include/cudaq/python/)

install(TARGETS cudaq-python-interop EXPORT cudaq-python-interop-targets DESTINATION lib)

install(EXPORT cudaq-python-interop-targets
        FILE CUDAQPythonInteropTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
