# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Fails when a public realtime header includes nvJitLink.h.  Building
# libcudaq-realtime needs the nvJitLink dev package; applications and bridges,
# which include only these headers, must not.
#
#   cmake -DINCLUDE_DIR=<realtime include dir> -P check_no_nvjitlink.cmake

if(NOT INCLUDE_DIR)
  message(FATAL_ERROR "INCLUDE_DIR must name the realtime include directory")
endif()
# CI's test jobs run ctest from the build tree alone, without the sources.
if(NOT IS_DIRECTORY "${INCLUDE_DIR}")
  message(STATUS "Skipped: no realtime headers at ${INCLUDE_DIR}")
  return()
endif()

file(GLOB_RECURSE headers
  "${INCLUDE_DIR}/*.h" "${INCLUDE_DIR}/*.hpp" "${INCLUDE_DIR}/*.cuh")
set(offenders)
foreach(header IN LISTS headers)
  file(STRINGS "${header}" hits REGEX "#[ \t]*include[ \t]*[<\"][^>\"]*nvJitLink")
  if(hits)
    list(APPEND offenders "${header}")
  endif()
endforeach()

if(offenders)
  list(JOIN offenders "\n  " listing)
  message(FATAL_ERROR "Public realtime headers include nvJitLink.h:\n  ${listing}")
endif()
list(LENGTH headers count)
message(STATUS "${count} public headers checked; none includes nvJitLink.h")
