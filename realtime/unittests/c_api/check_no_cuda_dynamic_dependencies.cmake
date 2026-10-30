# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT LIBRARY OR NOT EXISTS "${LIBRARY}")
  message(FATAL_ERROR "libcudaq-realtime does not exist: ${LIBRARY}")
endif()
if(NOT READELF)
  find_program(READELF readelf REQUIRED)
endif()

execute_process(
  COMMAND "${READELF}" -d "${LIBRARY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic
  ERROR_VARIABLE error)
if(result)
  message(FATAL_ERROR "readelf -d failed for ${LIBRARY}: ${error}")
endif()

foreach(dependency IN ITEMS libnvJitLink.so libcuda.so libcudart.so)
  if(dynamic MATCHES "Shared library: \\[${dependency}")
    message(FATAL_ERROR
      "libcudaq-realtime has a load-time dependency on ${dependency}:\n"
      "${dynamic}")
  endif()
endforeach()

execute_process(
  COMMAND "${READELF}" -Ws "${LIBRARY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE symbols
  ERROR_VARIABLE error)
if(result)
  message(FATAL_ERROR "readelf -Ws failed for ${LIBRARY}: ${error}")
endif()
if(symbols MATCHES "UND[^\n]*nvJitLink")
  message(FATAL_ERROR
    "libcudaq-realtime has an undefined nvJitLink symbol:\n${symbols}")
endif()

message(STATUS
  "libcudaq-realtime has no dynamic CUDA or nvJitLink dependency")
