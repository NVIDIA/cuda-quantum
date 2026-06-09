# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Create imported targets for test dependencies
if(NOT TARGET FileCheck)
  add_executable(FileCheck IMPORTED)
  set_target_properties(FileCheck PROPERTIES IMPORTED_LOCATION "/usr/local/llvm/bin/FileCheck")
endif()

llvm_canonicalize_cmake_booleans(
  CUDA_FOUND
)

# Each entry is "<target-basename>|<shim-source>".  Add a line here to register
# a new device-call realtime shim library; lit picks up the full file path and
# the test's `REQUIRES: device-call-realtime-lib` feature gate automatically.
set(CUDAQ_DEVICE_CALL_REALTIME_SHIMS
  "cudaq-test-device-call-realtime-scalar-gpu-dispatch|NVQPP/Inputs/device_call_realtime_scalar_shim.cu"
  "cudaq-test-device-call-realtime-scalar-host-dispatch|NVQPP/Inputs/device_call_realtime_scalar_host_dispatch_shim.cu"
  "cudaq-test-device-call-realtime-array-gpu-dispatch|NVQPP/Inputs/device_call_realtime_array_shim.cu"
  "cudaq-test-device-call-realtime-array-host-dispatch|NVQPP/Inputs/device_call_realtime_array_host_dispatch_shim.cu")

set(CUDAQ_DEVICE_CALL_REALTIME_LIBS "")
foreach(_shim IN LISTS CUDAQ_DEVICE_CALL_REALTIME_SHIMS)
  string(REPLACE "|" ";" _shim_parts "${_shim}")
  list(GET _shim_parts 0 _shim_target)
  list(APPEND CUDAQ_DEVICE_CALL_REALTIME_LIBS
    "${CMAKE_BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${_shim_target}${CMAKE_SHARED_LIBRARY_SUFFIX}")
endforeach()

if (CUDAQ_DISABLE_RUNTIME)
  set(CUDAQ_EXTRA_CONFIG_EXCLUDES "NVQPP")
endif()

add_subdirectory(plugin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
)

set(CUDAQ_TEST_PARAMS
  cudaq_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

get_property(test_cudaq_libraries GLOBAL PROPERTY CUDAQ_RUNTIME_LIBS)

set(CUDAQ_TEST_DEPENDS
    CircuitCheck
    cudaq-opt
    cudaq-rest-qpu
    cudaq-target-conf
    cudaq-translate
    CustomPassPlugin
    FileCheck
    nvq++
    test_argument_conversion
)

if (CUDA_FOUND AND CUDAQ_ENABLE_REALTIME AND NOT CUDAQ_DISABLE_CPP_FRONTEND AND
    NOT CUDAQ_DISABLE_RUNTIME)
  find_package(CUDAToolkit REQUIRED)

  function(add_cudaq_device_call_realtime_shim target source)
    add_library(${target} SHARED ${source})
    target_include_directories(${target}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/NVQPP/Inputs
        ${CMAKE_SOURCE_DIR}/runtime/include
        ${CMAKE_SOURCE_DIR}/runtime
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${target}
      PRIVATE
        cudaq-device-call-runtime
        cudaq-device-call-runtime-headers
        cudaq::cudaq-realtime-dispatch
        CUDA::cudart_static)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(${target}
      PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_STANDARD 20
        CXX_STANDARD 20
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  endfunction()

  foreach(_shim IN LISTS CUDAQ_DEVICE_CALL_REALTIME_SHIMS)
    string(REPLACE "|" ";" _shim_parts "${_shim}")
    list(GET _shim_parts 0 _shim_target)
    list(GET _shim_parts 1 _shim_source)
    add_cudaq_device_call_realtime_shim(
      ${_shim_target} ${CMAKE_CURRENT_SOURCE_DIR}/${_shim_source})
    list(APPEND CUDAQ_TEST_DEPENDS ${_shim_target})
  endforeach()
endif()

if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  set(CUDAQ_TEST_DEPENDS ${CUDAQ_TEST_DEPENDS}
    cudaq-quake
    fixup-linkage
    nvq++
    ${test_cudaq_libraries}
  )
endif()

add_custom_target(cudaq-test-depends DEPENDS ${CUDAQ_TEST_DEPENDS})
set_target_properties(cudaq-test-depends PROPERTIES FOLDER "Tests")

add_lit_testsuite(check-cudaq "Running the cudaq regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${CUDAQ_TEST_PARAMS}
  DEPENDS ${CUDAQ_TEST_DEPENDS}
)
set_target_properties(check-cudaq PROPERTIES FOLDER "Tools")

# In case of standalone builds.
if (CUDAQ_STANDALONE_BUILD)
  add_lit_testsuites(NVQPP ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${CUDAQ_TEST_PARAMS}
    DEPENDS ${CUDAQ_TEST_DEPENDS}
    FOLDER "Tests/Subdirectories"
  )
else()
  add_lit_testsuites(NVQPP ${CMAKE_CURRENT_SOURCE_DIR}
    PARAMS ${CUDAQ_TEST_PARAMS}
    DEPENDS ${CUDAQ_TEST_DEPENDS}
    FOLDER "Tests/Subdirectories"
  )
endif()

# Add cudaq tests to the ctest suite
add_test(NAME ctest-cudaq
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-cudaq
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
# This test runs llvm-lit internally with parallelism, so have ctest
# reserve all processors to prevent concurrent scheduling.
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 1)
endif()
set_tests_properties(ctest-cudaq PROPERTIES
  TIMEOUT 3600
  PROCESSORS ${NPROC})

# Setup an alias for 'check-all'.
add_custom_target(check)
add_dependencies(check check-all)
set_target_properties(check PROPERTIES FOLDER "Tests")
