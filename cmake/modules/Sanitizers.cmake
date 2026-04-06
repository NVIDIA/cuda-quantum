# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Sanitizers.cmake - Configure Address Sanitizer (ASan) and Undefined Behavior
# Sanitizer (UBSan) for memory error detection.
#
# This module is enabled by setting CUDAQ_ENABLE_SANITIZERS=ON.
#
# Sanitizer flags:
# -fsanitize=address: Detects use-after-free, buffer overflows, stack overflows
# -fsanitize=undefined: Detects undefined behavior (null ptr deref, signed overflow, etc.)
# -fno-sanitize=vptr: Disable vptr sanitizer (requires RTTI, but LLVM is built without RTTI)
# -fno-omit-frame-pointer: Preserves frame pointers for better stack traces
# -fno-optimize-sibling-calls: Disables tail call optimization for accurate stack traces
# -fsanitize-address-use-after-scope: Detects use-after-scope bugs (Clang only)

include_guard()

function(cudaq_enable_sanitizers)
  message(STATUS "Enabling Address Sanitizer (ASan) and Undefined Behavior Sanitizer (UBSan)...")
  
  # Warn if not using Debug build type
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Sanitizers are enabled but build type is '${CMAKE_BUILD_TYPE}'. "
                    "Consider using Debug build type for better stack traces and debug symbols.")
  endif()

  # Base sanitizer compile flags (as a list for add_compile_options)
  set(SANITIZER_COMPILE_FLAGS
    -fsanitize=address,undefined
    -fno-sanitize=vptr
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
  )
  
  # Add Clang-specific flag for use-after-scope detection
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND SANITIZER_COMPILE_FLAGS -fsanitize-address-use-after-scope)
  endif()

  # Sanitizer link flags
  set(SANITIZER_LINK_FLAGS -fsanitize=address,undefined)

  # Convert list to space-separated string for CMAKE_*_FLAGS variables
  # list(JOIN SANITIZER_COMPILE_FLAGS " " SANITIZER_COMPILE_FLAGS_STR)
  # list(JOIN SANITIZER_LINK_FLAGS " " SANITIZER_LINK_FLAGS_STR)

  message(STATUS "  Sanitizer compile flags: ${SANITIZER_COMPILE_FLAGS}")
  message(STATUS "  Sanitizer link flags: ${SANITIZER_LINK_FLAGS}")

  # Apply flags globally using add_compile_options and add_link_options
  # These affect all targets defined after this point
  add_compile_options(${SANITIZER_COMPILE_FLAGS})
  add_link_options(${SANITIZER_LINK_FLAGS})
endfunction()
