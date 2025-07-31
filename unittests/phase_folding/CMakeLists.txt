# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_opt OptTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_opt PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_opt
  PUBLIC . ${CMAKE_SOURCE_DIR}/runtime 
           ${CMAKE_SOURCE_DIR}/runtime/common 
           ${CMAKE_SOURCE_DIR}/runtime/nvqir
)
target_link_libraries(test_opt
  PRIVATE
    fmt::fmt-header-only
    cudaq
    cudaq-common
    cudaq-platform-mqpu
    nvqir
    nvqir-qpp
    gtest_main
)

function(generate_random_test template seed target_name)
  if (PYTHON_FOUND)
    add_custom_command(COMMAND ${Python_EXECUTABLE} random_gen.py ${template} --seed=${seed} > ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cpp
      OUTPUT ${target_name}.cpp
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(${target_name} ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cpp)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(${target_name} PRIVATE -Wl,--no-as-needed)
    endif()
    target_include_directories(${target_name}
      PUBLIC . ${CMAKE_SOURCE_DIR}/runtime 
              ${CMAKE_SOURCE_DIR}/runtime/common 
              ${CMAKE_SOURCE_DIR}/runtime/nvqir
    )
    target_link_libraries(${target_name}
      PRIVATE
        fmt::fmt-header-only
        cudaq
        cudaq-common
        cudaq-platform-mqpu
        nvqir
        nvqir-qpp
        gtest_main
    )
    gtest_discover_tests(${target_name})
  endif()
endfunction()

gtest_discover_tests(test_opt)
generate_random_test("testRNG0.template" "27" "test_rng0")
generate_random_test("testRNG0.template" "4" "test_rng01")
generate_random_test("testRNG1.template" "100" "test_rng1")
