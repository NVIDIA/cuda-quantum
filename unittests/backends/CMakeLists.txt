# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# List of libraries to link with by default to create a test executable
set(default_backend_unittest_libs
  fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-operator
  nvqir nvqir-qpp
  cudaq-platform-default 
  gtest_main)

define_property(DIRECTORY PROPERTY BACKEND_UNITTEST_LIBS INHERITED
  BRIEF_DOCS "Default libraries for backend unit tests"
  FULL_DOCS "List of libraries that are linked by default when \
creating backend unit test executables. Subdirectories can override \
this property to customize the default link libraries for their tests. \
Individual tests can also override this by passing LINK_LIBS to \
add_backend_unittest_executable()."
)
set_property(DIRECTORY PROPERTY BACKEND_UNITTEST_LIBS ${default_backend_unittest_libs})

# Helper function to create an executable to be used by the gtest unit tests
# - target: positional argument, name of the executable
# - BACKEND: named argument to specify a prefix for the test names
# - BACKEND_CONFIG: if present, the test will set NVQPP_TARGET_BACKEND_CONFIG 
#     with this value so the backend gets loaded by a constructor before entering main.
#     To avoid issues with semicolon the format is: backend key1=value1 key2=value2
#     The function will convert this to           : backend;key1;value1;key2;value2
#     Example:     infleqtion emulate=false url=http://localhost:62447
# - LINK_LIBS: optional argument to provide non-default list of libraries to link with 
function(add_backend_unittest_executable target)
  set(singleValues BACKEND BACKEND_CONFIG)
  set(multiValues SOURCES INCLUDES LINK_LIBS)
  cmake_parse_arguments(ARG "" "${singleValues}" "${multiValues}" ${ARGN})

  add_executable(${target} ${ARG_SOURCES})
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(${target} PRIVATE -Wl,--no-as-needed)
  endif()

  if(ARG_BACKEND)
    target_compile_definitions(${target} PRIVATE -DNVQIR_BACKEND_NAME=${ARG_BACKEND})
  endif()

  if (ARG_INCLUDES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDES})
  else()
    target_include_directories(${target} PRIVATE ../..)
  endif()

  if(ARG_LINK_LIBS)
    set(libs ${ARG_LINK_LIBS})
  else()
    set(libs ${default_backend_unittest_libs})
  endif()

  # Pass the configuration string to NVQPP_TARGET_BACKEND_CONFIG.
  if(ARG_BACKEND_CONFIG)
    string(REPLACE [[=]] [[\\;]] ARG_BACKEND_CONFIG "${ARG_BACKEND_CONFIG}")
    string(REPLACE [[ ]] [[\\;]] ARG_BACKEND_CONFIG "${ARG_BACKEND_CONFIG}")
    target_compile_definitions(${target} PRIVATE NVQPP_TARGET_BACKEND_CONFIG="${ARG_BACKEND_CONFIG}")
  endif()

  target_link_libraries(${target} PRIVATE ${libs})
endfunction()

find_package(Python COMPONENTS Interpreter)
if (OPENSSL_FOUND AND CUDAQ_ENABLE_PYTHON AND CUDAQ_TEST_MOCK_SERVERS)
  add_subdirectory(anyon)
  if (CUDAQ_ENABLE_BRAKET_BACKEND)
    add_subdirectory(braket)
  endif()
  add_subdirectory(infleqtion)
  add_subdirectory(ionq)
  add_subdirectory(iqm)
  add_subdirectory(oqc)
  if (CUDAQ_ENABLE_QCI_BACKEND)
    add_subdirectory(qci)
  endif()
  add_subdirectory(quantinuum)
  if (CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND)
    add_subdirectory(quantum_machines)
  endif()
  add_subdirectory(extra_payload_provider)
endif()
add_subdirectory(pasqal)
add_subdirectory(qpp_observe)
add_subdirectory(quera)
