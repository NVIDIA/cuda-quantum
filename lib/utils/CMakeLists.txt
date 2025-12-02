# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build utils library with conditional instrumentation based on backend selection
set(NVQLINK_UTILS_SOURCES ""
  #logger.cpp
)

# Add profiler backend sources
if(NVQLINK_PROFILER_BACKEND STREQUAL "TRACY" OR NVQLINK_PROFILER_BACKEND STREQUAL "NVTX")
  list(APPEND NVQLINK_UTILS_SOURCES instrumentation/nvtx.cpp)
endif()

# Add logging backend sources
if(NVQLINK_LOGGING_BACKEND STREQUAL "QUILL")
  list(APPEND NVQLINK_UTILS_SOURCES instrumentation/quill_logger.cpp)
endif()

add_library(nvqlink_utils STATIC ${NVQLINK_UTILS_SOURCES})

target_include_directories(nvqlink_utils
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_NVQLINK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(nvqlink_utils PUBLIC Threads::Threads)

#===============================================================================
# Profiler Backend Configuration
#===============================================================================

if(NVQLINK_PROFILER_BACKEND STREQUAL "TRACY")
  # Tracy profiler setup via FetchContent
  include(FetchContent)
  FetchContent_Declare(tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.13.0
    GIT_SHALLOW TRUE
  )
  
  # Tracy client library options (optimized for low-latency network daemon)
  set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiler" FORCE)
  set(TRACY_ON_DEMAND ON CACHE BOOL "Enable on-demand profiling (lower overhead when not connected)" FORCE)
  set(TRACY_NO_FRAME_IMAGE ON CACHE BOOL "Disable frame image capture (games only)" FORCE)
  set(TRACY_NO_VSYNC_CAPTURE ON CACHE BOOL "Disable vsync capture (games only)" FORCE)
  set(TRACY_NO_SAMPLING ON CACHE BOOL "Disable statistical sampling (use manual instrumentation)" FORCE)
  set(TRACY_CALLSTACK 0 CACHE STRING "Callstack depth (0=disabled, adds overhead)" FORCE)
  
  FetchContent_MakeAvailable(tracy)
  
  # Option to build Tracy profiler GUI
  option(BUILD_TRACY_PROFILER "Build Tracy profiler GUI (requires OpenSSL, GLFW3, etc.)" OFF)
  option(TRACY_LEGACY "Use legacy X11 backend for Tracy profiler (recommended for X11 systems)" ON)
  
  message(STATUS "Profiler backend: Tracy (v0.13.0)")
  message(STATUS "  Tracy client library: Built automatically")
  message(STATUS "  Configuration: On-Demand mode (connect profiler to start)")
  message(STATUS "  Optimizations: Game features disabled, manual instrumentation only")
  
  if(BUILD_TRACY_PROFILER AND NOT CMAKE_CROSSCOMPILING)
    FetchContent_GetProperties(tracy)
    if(tracy_POPULATED)
      # Set Tracy profiler windowing backend
      if(TRACY_LEGACY)
        set(LEGACY ON CACHE BOOL "Use legacy X11 backend for Tracy profiler" FORCE)
        message(STATUS "  Tracy windowing: Legacy X11")
      else()
        set(LEGACY OFF CACHE BOOL "Use Wayland backend for Tracy profiler" FORCE)
        message(STATUS "  Tracy windowing: Wayland")
      endif()
      
      if(EXISTS "${tracy_SOURCE_DIR}/profiler/CMakeLists.txt")
        add_subdirectory(${tracy_SOURCE_DIR}/profiler ${CMAKE_BINARY_DIR}/tracy-profiler EXCLUDE_FROM_ALL)
        
        message(STATUS "  Tracy profiler GUI: Enabled (target: tracy-profiler)")
        message(STATUS "")
        message(STATUS "To build Tracy profiler GUI:")
        message(STATUS "  cmake --build ${CMAKE_BINARY_DIR} --target tracy-profiler")
        message(STATUS "")
        if(TRACY_LEGACY)
          message(STATUS "Dependencies required (Legacy X11):")
          message(STATUS "  Core: libssl-dev libglfw3-dev libfreetype-dev libcapstone-dev")
          message(STATUS "  X11:  libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev")
          message(STATUS "")
          message(STATUS "Install all (Ubuntu/Debian):")
          message(STATUS "  sudo apt install libssl-dev libglfw3-dev libfreetype-dev libcapstone-dev \\")
          message(STATUS "                   libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev")
        else()
          message(STATUS "Dependencies required (Wayland): OpenSSL, GLFW3, FreeType, Capstone, xkbcommon, wayland")
          message(STATUS "  Ubuntu/Debian: sudo apt install libssl-dev libglfw3-dev libfreetype-dev libcapstone-dev libxkbcommon-dev libwayland-dev")
        endif()
      else()
        message(WARNING "Tracy profiler CMakeLists.txt not found")
        message(WARNING "Tracy profiler GUI will not be available")
      endif()
    endif()
  else()
    message(STATUS "  Tracy profiler GUI: Disabled (use -DBUILD_TRACY_PROFILER=ON to enable)")
    message(STATUS "")
    message(STATUS "Or download pre-built Tracy profiler:")
    message(STATUS "  https://github.com/wolfpld/tracy/releases/download/v0.13.0/Tracy-0.13.0.7z")
  endif()
  
  # Link Tracy to nvqlink_utils
  target_link_libraries(nvqlink_utils PUBLIC TracyClient)
  target_compile_definitions(nvqlink_utils PUBLIC PROFILER_BACKEND_TRACY TRACY_ENABLE)
  
elseif(NVQLINK_PROFILER_BACKEND STREQUAL "NVTX")
  # NVTX profiler setup
  if(CUDA_FOUND)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(nvqlink_utils PUBLIC CUDA::nvToolsExt)
    target_compile_definitions(nvqlink_utils PUBLIC PROFILER_BACKEND_NVTX)
    message(STATUS "Profiler backend: NVTX/Nsight")
  else()
    message(WARNING "NVTX profiler requested but CUDA not found, disabling profiler")
  endif()
  
else()
  message(STATUS "Profiler backend: NONE (instrumentation disabled)")
endif()

#===============================================================================
# Logging Backend Configuration
#===============================================================================

if(NVQLINK_LOGGING_BACKEND STREQUAL "QUILL")
  # Quill logger setup via FetchContent
  include(FetchContent)
  FetchContent_Declare(quill
    GIT_REPOSITORY https://github.com/odygrd/quill.git
    GIT_TAG v11.0.1
    GIT_SHALLOW TRUE
  )
  
  # Quill library options
  set(QUILL_BUILD_EXAMPLES OFF CACHE BOOL "Build Quill examples" FORCE)
  set(QUILL_BUILD_TESTS OFF CACHE BOOL "Build Quill tests" FORCE)
  
  FetchContent_MakeAvailable(quill)
  
  # Link Quill to nvqlink_utils
  target_link_libraries(nvqlink_utils PUBLIC quill::quill)
  target_compile_definitions(nvqlink_utils PUBLIC LOGGING_BACKEND_QUILL)
  
  # Compile-time log level filtering
  if(NVQLINK_LOGGING_LEVEL STREQUAL "TRACE")
    target_compile_definitions(nvqlink_utils PUBLIC QUILL_ACTIVE_LOG_LEVEL=QUILL_LOG_LEVEL_TRACE_L3)
  elseif(NVQLINK_LOGGING_LEVEL STREQUAL "DEBUG")
    target_compile_definitions(nvqlink_utils PUBLIC QUILL_ACTIVE_LOG_LEVEL=QUILL_LOG_LEVEL_DEBUG)
  elseif(NVQLINK_LOGGING_LEVEL STREQUAL "INFO")
    target_compile_definitions(nvqlink_utils PUBLIC QUILL_ACTIVE_LOG_LEVEL=QUILL_LOG_LEVEL_INFO)
  elseif(NVQLINK_LOGGING_LEVEL STREQUAL "WARNING")
    target_compile_definitions(nvqlink_utils PUBLIC QUILL_ACTIVE_LOG_LEVEL=QUILL_LOG_LEVEL_WARNING)
  elseif(NVQLINK_LOGGING_LEVEL STREQUAL "ERROR")
    target_compile_definitions(nvqlink_utils PUBLIC QUILL_ACTIVE_LOG_LEVEL=QUILL_LOG_LEVEL_ERROR)
  endif()
  
  message(STATUS "Logging backend: Quill v11.0.1 (level: ${NVQLINK_LOGGING_LEVEL})")
else()
  message(STATUS "Logging backend: NONE (logging disabled)")
endif()
