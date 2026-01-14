# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Fetch spdlog from its GitHub repository
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.15.0           # Update the tag to the version you want
)
FetchContent_MakeAvailable(spdlog)

set(LIBRARY_NAME cudaq-nvqlink)

add_compile_options(-Wno-attributes) 

# FIXME?: This must be a shared library. Trying to build a static one will fail.
add_library(${LIBRARY_NAME} SHARED
  nvqlink.cpp
  compiler.cpp
  utils/logger.cpp
)

add_subdirectory(compilers)

target_include_directories(${LIBRARY_NAME}
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQ_NVQLINK_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${spdlog_SOURCE_DIR}/include/spdlog>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBRARY_NAME} PRIVATE spdlog::spdlog)

set_target_properties(${LIBRARY_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

install(TARGETS ${LIBRARY_NAME}
  COMPONENT nvqlink-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
    
install(DIRECTORY ${CUDAQ_NVQLINK_INCLUDE_DIR}/cudaq
  COMPONENT nvqlink-headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)

add_subdirectory(daemon)
add_subdirectory(network)
add_subdirectory(qcs)
add_subdirectory(utils)

# FIXME:
# add_subdirectory(plugins)