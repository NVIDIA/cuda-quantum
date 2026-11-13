# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The standalone checker is also usable with third-party provider libraries.
include_directories(${CUDAQ_REALTIME_SOURCE_DIR}/validation)
add_executable(playback_provider_contract playback_provider_contract.cpp)
target_include_directories(playback_provider_contract PRIVATE ${CUDAQ_REALTIME_INCLUDE_DIR})
target_compile_features(playback_provider_contract PRIVATE cxx_std_17)
target_link_libraries(playback_provider_contract PRIVATE ${CMAKE_DL_LIBS})

# Unit checks share one fixture and use the real provider libraries.
add_executable(test_playback_provider test_playback_provider.cpp)
target_include_directories(test_playback_provider PRIVATE ${CUDAQ_REALTIME_INCLUDE_DIR})
target_compile_features(test_playback_provider PRIVATE cxx_std_17)
target_link_libraries(test_playback_provider PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS})
target_compile_definitions(test_playback_provider PRIVATE
  UDP_PLAYBACK_PROVIDER_PATH="$<TARGET_FILE:cudaq-playback-provider-udp>")
add_dependencies(test_playback_provider cudaq-playback-provider-udp)
if (TARGET cudaq-playback-provider-hsb)
  target_compile_definitions(test_playback_provider PRIVATE
    HSB_PLAYBACK_PROVIDER_PATH="$<TARGET_FILE:cudaq-playback-provider-hsb>")
  add_dependencies(test_playback_provider cudaq-playback-provider-hsb)
endif()
cudaq_gtest_discover_tests(test_playback_provider
  TEST_PREFIX "test_playback_provider." PROPERTIES TIMEOUT 30)

# Python is needed only when CTest runs these drivers, not to configure,
# build or use the installed validation apps. Resolve it on the test host.
configure_file(test_playback.py test_playback.py COPYONLY)
configure_file(test_runner_options.py test_runner_options.py COPYONLY)
# Stub processes test argument forwarding and defaults without networking.
add_test(NAME validation_runner_options
  COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_CURRENT_BINARY_DIR}/test_runner_options.py
    --runner ${CUDAQ_REALTIME_VALIDATION_BINARY_DIR}/validation_test.sh
    --legacy-runner ${CUDAQ_REALTIME_VALIDATION_BINARY_DIR}/roce_test.sh)
set_tests_properties(validation_runner_options PROPERTIES TIMEOUT 30)
# An independent UDP socket peer checks transmitted bytes and injects delayed,
# malformed or missing replies to exercise capture and reply verification.
foreach(case IN ITEMS paced corrupt omit delayed late-duplicate)
  add_test(NAME playback_udp_peer_${case}
    COMMAND ${CMAKE_COMMAND} -E env python3
      ${CMAKE_CURRENT_BINARY_DIR}/test_playback.py
      --playback $<TARGET_FILE:playback_app>
      --plugin $<TARGET_FILE:cudaq-playback-provider-udp> --fixture ${case})
  set_tests_properties(playback_udp_peer_${case} PROPERTIES TIMEOUT 30)
endforeach()
# Installer CI covers separate/unified UDP runs; FPGA CI covers the RoCE modes.
# Run the provider contract checker against a real bridge to check input
# ownership and repeat retrieval across two batches.
if (TARGET validation_app AND TARGET cudaq-realtime-bridge-udp)
  add_test(NAME validation_app_udp_contract
    COMMAND ${CMAKE_COMMAND} -E env python3
      ${CMAKE_CURRENT_BINARY_DIR}/test_playback.py
      --plugin $<TARGET_FILE:cudaq-playback-provider-udp>
      --contract $<TARGET_FILE:playback_provider_contract>
      --consumer $<TARGET_FILE:validation_app>
      --bridge $<TARGET_FILE:cudaq-realtime-bridge-udp>)
  set_tests_properties(validation_app_udp_contract PROPERTIES TIMEOUT 30)
endif()
