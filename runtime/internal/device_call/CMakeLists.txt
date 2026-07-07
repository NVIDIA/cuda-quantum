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
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
  $<INSTALL_INTERFACE:include>)

if (CUDA_FOUND AND CUDAQ_ENABLE_REALTIME)
  target_link_libraries(cudaq-device-call-runtime-headers INTERFACE
    cudaq::cudaq-realtime)

  set(CUDAQ_DEVICE_CALL_SOURCES
      DeviceCallDispatch.cpp
      DeviceCallService.cpp
      DeviceCallChannel.cpp
      GpuDispatchChannel.cpp
      HostDispatchChannel.cpp
      RingBufferWrapper.cpp
      DeviceCallRuntimePlugin.cpp
      RpcFrame.cpp)

  # Optional CPU-RoCE RDMA channel (cpu_transport Phase 2). Only available when
  # the realtime package was built with libibverbs (i.e. the cpu-transport
  # target exists); hosts without RDMA simply omit the cpu_roce channel.
  set(CUDAQ_DEVICE_CALL_CPU_ROCE_LIB "")
  if (TARGET cudaq::cudaq-realtime-cpu-transport)
    list(APPEND CUDAQ_DEVICE_CALL_SOURCES CpuRoceChannel.cpp)
    set(CUDAQ_DEVICE_CALL_CPU_ROCE_LIB cudaq::cudaq-realtime-cpu-transport)
    message(STATUS "  - device_call: cpu_roce channel enabled (libibverbs)")
  endif()

  # UDP channel: device_call over plain UDP for systems without an RDMA NIC,
  # over loopback or a real network.
  set(CUDAQ_DEVICE_CALL_UDP_LIB "")
  if (TARGET cudaq::cudaq-realtime-udp-transport)
    list(APPEND CUDAQ_DEVICE_CALL_SOURCES UdpChannel.cpp)
    set(CUDAQ_DEVICE_CALL_UDP_LIB cudaq::cudaq-realtime-udp-transport)
    message(STATUS "  - device_call: udp channel enabled")
  endif()

  add_library(cudaq-device-call-runtime
    SHARED
      ${CUDAQ_DEVICE_CALL_SOURCES})

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
      ${CUDAQ_DEVICE_CALL_CPU_ROCE_LIB}
      ${CUDAQ_DEVICE_CALL_UDP_LIB}
      LLVMSupport
      CUDA::cudart_static
      dl)
  target_link_options(cudaq-device-call-runtime PRIVATE ${CUDAQ_FORCE_LINK_FLAG})

  install(TARGETS cudaq-device-call-runtime-headers
          EXPORT cudaq-targets)

  install(TARGETS cudaq-device-call-runtime
          EXPORT cudaq-targets
          DESTINATION lib)

  install(FILES
          ${CMAKE_SOURCE_DIR}/runtime/cudaq/realtime/device_call_service.h
          DESTINATION include/cudaq/realtime)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION include)
endif()
