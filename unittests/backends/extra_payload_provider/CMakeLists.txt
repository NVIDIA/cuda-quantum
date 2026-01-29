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


add_backend_unittest_executable(test_fake_extra_payload_provider 
  SOURCES FakeHorizonTester.cpp
  BACKEND horizon
  BACKEND_CONFIG "horizon emulate=false extra_payload_provider=sunrise"
)

configure_file(HorizonStartServerAndTest.sh.in HorizonStartServerAndTest.sh @ONLY)
add_test(NAME extra-payload-provider-tests COMMAND bash HorizonStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

