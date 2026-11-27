# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# An application's own unified dispatch kernel
# ==============================================================================
# Supported forms:
#
#   cudaq_realtime_add_dispatch_kernel(<target> <source>)
#   cudaq_realtime_add_dispatch_kernel(<target> <source> ARCH <N>)
#
# For example:
#
#   find_package(cudaq-realtime CONFIG REQUIRED)
#   add_executable(my_app main.cpp handlers.cu)
#   set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#   target_link_libraries(my_app PRIVATE cudaq::cudaq-realtime)
#   cudaq_realtime_add_dispatch_kernel(my_app my_dispatcher.cu)
#
# Or choose the LTO-IR architecture explicitly:
#
#   cudaq_realtime_add_dispatch_kernel(my_app my_dispatcher.cu ARCH 90)
#
# <target> must already exist in the calling directory.  <source> may be
# relative to that directory or absolute, and must not be one of <target>'s
# ordinary sources.  <N> is a numeric CUDA architecture such as 80, 90 or 100.
#
# Adds the dispatch kernels of the CUDA file <source> to <target>, the binary
# that also holds the DEVICE_CALL handlers.  A kernel there is
# `extern "C" __global__`, calls the transport hooks of
# unified_device_transport.cuh, and is found at run time with
# cudaq_dispatcher_get_device_kernel; the application's launcher
# (cudaq_dispatcher_set_unified_launch) starts it.
#
# Do not list <source> among the target's own sources.  It calls the bridge's
# hooks, which nothing in the application defines, so the target's device link
# would fail on them.  Instead it is compiled on its own to LTO-IR, as
# libcudaq-realtime does for its own kernel, and embedded in the target's
# __nv_relfatbin section, where cudaq_device_code_of_this_binary hands it to
# the dispatcher with the handlers.  The kernel sees the target's include
# directories, compile definitions and CUDA_STANDARD (default 17).
#
# ARCH is the GPU architecture of the LTO-IR, by default the one
# libcudaq-realtime's kernel is built for
# (CUDAQ_REALTIME_DEVICE_CODE_MIN_ARCH): LTO-IR for one architecture links for
# that GPU and every newer one.
#
# Call it from the directory that defines <target>: the source properties it
# sets belong to the calling directory.  See
# realtime/docs/device_code_contract.md, "Providing your own dispatch kernel".

function(cudaq_realtime_add_dispatch_kernel target source)
  cmake_parse_arguments(PARSE_ARGV 2 _arg "" "ARCH" "")
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "cudaq_realtime_add_dispatch_kernel(${target}): "
      "CUDA is not enabled; call enable_language(CUDA) first")
  endif()
  if (NOT TARGET ${target})
    message(FATAL_ERROR
      "cudaq_realtime_add_dispatch_kernel: no target '${target}'")
  endif()
  get_target_property(_target_binary_dir ${target} BINARY_DIR)
  if (NOT "${_target_binary_dir}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
    message(FATAL_ERROR
      "cudaq_realtime_add_dispatch_kernel(${target}): call the function from "
      "the directory that defines the target (${_target_binary_dir})")
  endif()
  set(_arch ${_arg_ARCH})
  if (NOT _arch)
    set(_arch ${CUDAQ_REALTIME_DEVICE_CODE_MIN_ARCH})
  endif()
  if (NOT _arch)
    set(_arch 80)
  endif()

  get_filename_component(_source ${source} ABSOLUTE)
  get_filename_component(_name ${source} NAME_WE)
  set(_fatbin ${CMAKE_CURRENT_BINARY_DIR}/${target}_${_name}.fatbin)
  set(_embed ${CMAKE_CURRENT_BINARY_DIR}/${target}_${_name}_image.cpp)

  set(_nvcc_flags)
  if (CMAKE_CUDA_HOST_COMPILER)
    list(APPEND _nvcc_flags -ccbin ${CMAKE_CUDA_HOST_COMPILER})
  endif()
  # The realtime headers normally come with the target's link to
  # cudaq::cudaq-realtime; this also covers a target that does not link it.
  if (CUDAQ_REALTIME_INCLUDE_DIR)
    list(APPEND _nvcc_flags -I${CUDAQ_REALTIME_INCLUDE_DIR})
  endif()
  set(_includes "$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>")
  set(_defines "$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>")
  set(_standard "$<TARGET_PROPERTY:${target},CUDA_STANDARD>")
  # Evaluate language-guarded target properties once per target language, then
  # give nvcc the CUDA result.
  set(_response
    ${CMAKE_CURRENT_BINARY_DIR}/${target}_${_name}_$<CONFIG>_$<COMPILE_LANGUAGE>.rsp)
  set(_cuda_response
    ${CMAKE_CURRENT_BINARY_DIR}/${target}_${_name}_$<CONFIG>_CUDA.rsp)
  set(_quote "\"")
  set(_escaped_quote "\\\\\"")
  set(_escaped_defines
    "$<LIST:TRANSFORM,${_defines},REPLACE,${_quote},${_escaped_quote}>")
  string(CONCAT _response_content
    "$<$<BOOL:${_includes}>:-I\"$<JOIN:${_includes},\"\n-I\">\">\n"
    "$<$<BOOL:${_defines}>:-D$<JOIN:${_escaped_defines},\n-D>>\n")
  file(GENERATE
    OUTPUT ${_response}
    CONTENT "${_response_content}"
    TARGET ${target})
  add_custom_command(
    OUTPUT ${_fatbin}
    COMMAND ${CMAKE_CUDA_COMPILER} ${_nvcc_flags}
      -fatbin -rdc=true
      "-std=c++$<IF:$<BOOL:${_standard}>,${_standard},17>"
      --generate-code=arch=compute_${_arch},code=lto_${_arch}
      --options-file ${_cuda_response}
      -MD -MF ${_fatbin}.d -MT ${_fatbin}
      ${_source} -o ${_fatbin}
    DEPENDS ${_source} ${_cuda_response}
    DEPFILE ${_fatbin}.d
    COMMENT "Building the LTO-IR image of ${_name} for ${target}"
    COMMAND_EXPAND_LISTS
    VERBATIM)
  configure_file(
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cudaq_realtime_dispatch_kernel_image.cpp.in
    ${_embed} @ONLY)
  set_source_files_properties(${_embed} PROPERTIES OBJECT_DEPENDS ${_fatbin})
  target_sources(${target} PRIVATE ${_embed})
endfunction()
