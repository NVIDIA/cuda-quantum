# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build/install the generic RTH receiver and its shared GPU handler support.
# CUDA is needed to build the executable even when it will run CPU dispatch;
# networking libraries are selected at runtime through the bridge API.
# The universal receiver depends on the dispatcher, not on RoCE or HSB.
if (TARGET cudaq-realtime-dispatch AND CMAKE_CUDA_COMPILER)
  add_library(rpc_increment_ft STATIC
    ${CMAKE_CURRENT_LIST_DIR}/init_rpc_increment_function_table.cu)
  # Keep device code relocatable until the final executable is device-linked,
  # where function-table initialization can resolve the handler's address.
  set_target_properties(rpc_increment_ft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 20)
  target_include_directories(rpc_increment_ft PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  # Each executable needs the exported handler in its own CUDA module; an
  # unreferenced archive member would otherwise be omitted by device linking.
  target_sources(rpc_increment_ft INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/rpc_increment_handler.cu)

  # Compile both host and device handlers into one receiver executable.
  add_executable(validation_app
    ${CMAKE_CURRENT_LIST_DIR}/validation_app.cpp
    ${CMAKE_CURRENT_LIST_DIR}/init_rpc_increment_function_table_host.cpp)
  # Locate installed runtime libraries relative to bin/, without requiring
  # LD_LIBRARY_PATH. Resolve device symbols for the GPU function table.
  set_target_properties(validation_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_RPATH "$ORIGIN/../lib"
    LINKER_LANGUAGE CUDA CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(validation_app PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(validation_app PRIVATE
    cudaq-realtime-dispatch cudaq-realtime rpc_increment_ft Threads::Threads)
  # Build whichever bridges are available, without linking them into the app
  # or requiring SDKs for bridges omitted from this build.
  foreach(provider IN ITEMS udp cpu-roce gpu-roce)
    if (TARGET cudaq-realtime-bridge-${provider})
      add_dependencies(validation_app cudaq-realtime-bridge-${provider})
    endif()
  endforeach()
  install(TARGETS validation_app DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT realtime-tests)
endif()
