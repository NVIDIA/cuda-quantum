# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Applications, playback providers and runners shipped for installation
# validation. Unit tests and synthetic providers live in unittests/validation.
# Build the common handler first; both the generic and RoCE apps use it.
include(validation_app.cmake)
if (CUDAQ_REALTIME_ENABLE_HSB_TOOLS)
  include(hsb_tools.cmake)
endif()

# Copy runners so tests and installation need only build artifacts.
# The legacy CLI is also tested when HSB tools are disabled.
configure_file(roce_test.sh roce_test.sh COPYONLY)
configure_file(validation_test.sh validation_test.sh COPYONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/validation_test.sh
  DESTINATION utils COMPONENT realtime-tests)

# The standalone playback app and UDP provider need no CUDA or HSB SDK.
# Playback loads providers with dlopen; UDP uses a worker thread for I/O.
find_package(Threads REQUIRED)
add_executable(playback_app playback_app.cpp)
add_library(cudaq-playback-provider-udp SHARED udp_playback_provider.cpp)
foreach(target IN ITEMS playback_app cudaq-playback-provider-udp)
  target_include_directories(${target} PRIVATE ${CUDAQ_REALTIME_INCLUDE_DIR})
  # The provider getter is looked up by name at runtime, so preserve exported
  # C symbols even when a parent build defaults to hidden C++ visibility.
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON CXX_VISIBILITY_PRESET default)
endforeach()
target_link_libraries(playback_app PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(cudaq-playback-provider-udp PRIVATE Threads::Threads)
install(TARGETS playback_app cudaq-playback-provider-udp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT realtime-tests
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT realtime-tests)

# Build the HSB provider only alongside the optional direct playback tool,
# reusing that tool's SDK dependencies.
if (TARGET hsb_fpga_playback)
  add_library(cudaq-playback-provider-hsb SHARED hsb_playback_provider.cpp)
  target_include_directories(cudaq-playback-provider-hsb PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR} ${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS})
  # Reuse the direct playback target's SDK include and link settings.
  get_target_property(_hsb_includes hsb_fpga_playback INCLUDE_DIRECTORIES)
  get_target_property(_hsb_links hsb_fpga_playback LINK_LIBRARIES)
  target_include_directories(cudaq-playback-provider-hsb PRIVATE ${_hsb_includes})
  target_link_libraries(cudaq-playback-provider-hsb PRIVATE ${_hsb_links})
  set_target_properties(cudaq-playback-provider-hsb PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_VISIBILITY_PRESET default)
  install(TARGETS cudaq-playback-provider-hsb
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT hsb-tools)
endif()
