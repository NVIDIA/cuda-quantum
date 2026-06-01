# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-device-call-runtime-headers INTERFACE)
target_include_directories(cudaq-device-call-runtime-headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if (CUDA_FOUND AND CUDAQ_ENABLE_REALTIME)
  target_link_libraries(cudaq-device-call-runtime-headers INTERFACE
    cudaq::cudaq-realtime)

  add_library(cudaq-device-call-runtime
    SHARED
      DeviceCallDispatch.cpp
      DeviceCallService.cpp
      DeviceCallChannel.cpp
      GpuDispatchChannel.cpp
      HostDispatchChannel.cpp
      RingBufferWrapper.cpp
      DeviceCallRuntimePlugin.cpp
      RpcFrame.cpp)

  set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS
    cudaq-device-call-runtime)

  target_include_directories(cudaq-device-call-runtime
    PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime
      ${CMAKE_SOURCE_DIR}/runtime/include)

  target_link_libraries(cudaq-device-call-runtime
    PUBLIC
      cudaq-device-call-runtime-headers
    PRIVATE
      cudaq-common
      cudaq::cudaq-realtime
      LLVMSupport
      CUDA::cudart_static
      dl)
  target_link_options(cudaq-device-call-runtime PRIVATE ${CUDAQ_FORCE_LINK_FLAG})

  install(TARGETS cudaq-device-call-runtime-headers
          EXPORT cudaq-targets)

  install(TARGETS cudaq-device-call-runtime
          EXPORT cudaq-targets
          DESTINATION lib)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION include)
endif()
