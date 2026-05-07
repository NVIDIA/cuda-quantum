# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message(STATUS "Building cudaq-pyscf plugin.")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins)
if(APPLE)
  SET(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..")
else()
  SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
endif()
add_library(cudaq-pyscf SHARED PySCFDriver.cpp)

target_compile_options(cudaq-pyscf PRIVATE -Wno-cast-qual)

target_include_directories(cudaq-pyscf PRIVATE
    ${Python3_INCLUDE_DIRS}
)
if (SKBUILD)
  target_link_libraries(cudaq-pyscf
    PRIVATE
      nanobind-static Python3::Module
      cudaq-chemistry cudaq-operator cudaq cudaq-py-utils cudaq-platform-default)
  # Apple's linker (ld64) doesn't support --unresolved-symbols flag
  if (NOT APPLE)
    target_link_options(cudaq-pyscf
      PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)
  endif()
else()
  target_link_libraries(cudaq-pyscf
    PRIVATE
      nanobind-static Python3::Python
      cudaq-chemistry cudaq-operator cudaq cudaq-py-utils cudaq-platform-default)
endif()

install(TARGETS cudaq-pyscf DESTINATION lib/plugins)
