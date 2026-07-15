# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE FakeQuakeServerHelper.cpp) 
add_target_config(quake_fake)

add_library(cudaq-serverhelper-quake_fake SHARED FakeQuakeServerHelper.cpp)
target_link_libraries(cudaq-serverhelper-quake_fake PUBLIC cudaq-common fmt::fmt-header-only)
set_target_properties(cudaq-serverhelper-quake_fake PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
install(TARGETS cudaq-serverhelper-quake_fake DESTINATION lib)

add_library(fake_quake_backend_device_funcs SHARED backend_lib.cpp)
configure_file(QuakeStartServerAndTest.sh.in QuakeStartServerAndTest.sh @ONLY)

add_test(NAME quake-remote-backend-tests
  COMMAND bash QuakeStartServerAndTest.sh mock
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME quake-emulation-backend-tests
  COMMAND bash QuakeStartServerAndTest.sh emulation
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Account for the concurrent `nvq++` and Python processes in CTest's scheduler.
set_tests_properties(quake-remote-backend-tests PROPERTIES PROCESSORS 6)
set_tests_properties(quake-emulation-backend-tests PROPERTIES PROCESSORS 5)
