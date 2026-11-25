# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The runtime C API for nvq++: functions called by nvq++-generated code.
#
# - libcudaq-nvqpp-capi.a: linked into executables and shared libraries
#   produced by nvq++.
# - libcudaq-nvqpp-capi.{so,dylib}: loaded into the global scope by the
#   Python frontend, so that libraries built with `nvq++ -c` can be loaded.
#
# Both only export the functions declared with `CUDAQ_NVQPP_API` in
# nvqpp_capi.h.

cudaq_extract_exported_symbols(${CMAKE_CURRENT_SOURCE_DIR}/nvqpp_capi.h
  CAPI_SYMBOLS CUDAQ_NVQPP_API)

add_library(cudaq-nvqpp-capi-obj OBJECT nvqpp_capi.cpp)
set_target_properties(cudaq-nvqpp-capi-obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden)
target_link_libraries(cudaq-nvqpp-capi-obj PRIVATE cudaq cudaq-common)

# Shared library, for the Python frontend.
add_library(cudaq-nvqpp-capi SHARED $<TARGET_OBJECTS:cudaq-nvqpp-capi-obj>)
target_link_libraries(cudaq-nvqpp-capi PRIVATE cudaq cudaq-common)
cudaq_export_only_symbols(cudaq-nvqpp-capi ${CAPI_SYMBOLS})
if(NOT APPLE)
  target_link_options(cudaq-nvqpp-capi PRIVATE "LINKER:--no-undefined")
endif()
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-nvqpp-capi)
install(TARGETS cudaq-nvqpp-capi DESTINATION lib COMPONENT Runtime)

# Static archive, for nvq++. Same output name as the shared library.
add_library(cudaq-nvqpp-capi-static STATIC
  $<TARGET_OBJECTS:cudaq-nvqpp-capi-obj>)
set_target_properties(cudaq-nvqpp-capi-static PROPERTIES
  OUTPUT_NAME cudaq-nvqpp-capi
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
install(TARGETS cudaq-nvqpp-capi-static DESTINATION lib COMPONENT Nvqpp)

if(CUDAQ_BUILD_TESTS)
  add_test(NAME nvqpp_capi_exports
    COMMAND ${CMAKE_COMMAND}
      -DNM=${CMAKE_NM}
      -DCXX=${CMAKE_CXX_COMPILER}
      -DCAPI=$<TARGET_FILE:cudaq-nvqpp-capi>
      -DCAPI_STATIC=$<TARGET_FILE:cudaq-nvqpp-capi-static>
      -DRUNTIME=$<TARGET_FILE:cudaq>
      "-DEXPORTS=${CAPI_SYMBOLS}"
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/test
      -DAPPLE=${APPLE}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/CheckCApi.cmake)
endif()
