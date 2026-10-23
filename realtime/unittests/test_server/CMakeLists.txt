# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Transport-agnostic two-process test server
# ==============================================================================
# Service half of the realtime two-process tests: a dispatcher serving
# rpc_increment over whichever bridge provider --transport= names, as a
# HOST_CALL handler on the host (--path=host) or a DEVICE_CALL handler on the
# GPU (--path=device).  Not tied to any one transport, hence its own directory
# rather than living under bridge_interface/<transport>/.

# The increment handler + function-table initialiser is shared with
# hsb_bridge_cpu, where it is compiled BELOW that directory's CPU RoCE return()
# guard.  Referencing the source directly keeps this target independent of
# the CPU RoCE transport being built.
add_executable(cudaq-realtime-test-server
  cudaq_realtime_test_server.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_transport/init_rpc_increment_function_table_host.cpp
)

# CUDA headers only (no CUDA link): cudaq_realtime.h transitively includes
# <cuda_runtime.h> for the cudaGraphExec_t typedef in the function-entry union.
target_include_directories(cudaq-realtime-test-server PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# Where to look for provider .so's named by --transport=<name>, so the tests
# resolve them out of the build tree without an install.  Both shipped
# providers set LIBRARY_OUTPUT_DIRECTORY to this directory.
target_compile_definitions(cudaq-realtime-test-server PRIVATE
  CUDAQ_REALTIME_BRIDGE_PROVIDER_DIR="${CMAKE_BINARY_DIR}/lib"
)

target_link_libraries(cudaq-realtime-test-server PRIVATE
  cudaq-realtime
  cudaq-realtime-dispatch
  Threads::Threads
)

# --path=device needs the device increment handler, which is built only with
# the gpu_roce tooling (unittests/utils, configured before this directory).
# The dispatch kernels in libcudaq-realtime-dispatch.a and the handler are
# device-linked into this binary, so they share one CUDA module.  The
# CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH property is what the device-path
# tests gate on.
if (TARGET rpc_increment_ft)
  target_compile_definitions(cudaq-realtime-test-server PRIVATE
    CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH
  )
  set_target_properties(cudaq-realtime-test-server PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH ON
  )
  target_link_libraries(cudaq-realtime-test-server PRIVATE rpc_increment_ft)
  message(STATUS "  - cudaq-realtime-test-server: --path=device enabled")
endif()

# Providers are dlopen'd, not linked, so express the ordering explicitly for
# whichever ones this configure builds.
foreach(provider cudaq-realtime-bridge-udp cudaq-realtime-bridge-cpu-roce
                 cudaq-realtime-bridge-gpu-roce)
  if (TARGET ${provider})
    add_dependencies(cudaq-realtime-test-server ${provider})
  endif()
endforeach()

message(STATUS "  - cudaq-realtime-test-server (two-process dispatch server)")
