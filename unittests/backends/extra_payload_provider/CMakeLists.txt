# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE HorizonServerHelper.cpp)
add_target_config(horizon)

add_library(cudaq-serverhelper-horizon SHARED HorizonServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-horizon
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
set_target_properties(cudaq-serverhelper-horizon
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

install(TARGETS cudaq-serverhelper-horizon DESTINATION lib)

add_library(sunrise-provider SHARED ProviderLib.cpp)
target_link_libraries(sunrise-provider
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)


add_executable(test_fake_extra_payload_provider FakeHorizonTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_fake_extra_payload_provider PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_compile_definitions(test_fake_extra_payload_provider PRIVATE -DNVQIR_BACKEND_NAME=horizon)
target_include_directories(test_fake_extra_payload_provider PRIVATE ../..)
target_link_libraries(test_fake_extra_payload_provider
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-operator
  nvqir nvqir-qpp
  cudaq-platform-default 
  gtest_main)


configure_file("HorizonStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/extra_payload_provider/HorizonStartServerAndTest.sh" @ONLY)
add_test(NAME extra-payload-provider-tests COMMAND bash HorizonStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/extra_payload_provider/)

