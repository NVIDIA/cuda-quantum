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

function(generate_random_test template seed target_name block_length)
  if (PYTHON_FOUND)
    add_custom_command(COMMAND ${Python_EXECUTABLE} random_gen.py ${template} --seed=${seed} --block-length=${block_length} > ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cpp
      OUTPUT ${target_name}.cpp
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS random_gen.py ${template})
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

generate_random_test("straight-line.template" "27" "straight-line" "100")
generate_random_test("subkernel.template" "1" "subkernel-1" "15-25")

generate_random_test("branch-in-loop.template" "1" "branch-in-loop-1" "10-15")
generate_random_test("branch-in-loop.template" "2" "branch-in-loop-2" "10-15")
generate_random_test("branch-in-loop.template" "3" "branch-in-loop-3" "10-15")
generate_random_test("branch-in-loop.template" "4" "branch-in-loop-4" "10-15")
generate_random_test("branch-in-loop.template" "5" "branch-in-loop-5" "10-15")
generate_random_test("branch-in-loop.template" "6" "branch-in-loop-6" "10-15")
generate_random_test("branch-in-loop.template" "7" "branch-in-loop-7" "10-15")
generate_random_test("branch-in-loop.template" "8" "branch-in-loop-8" "10-15")
generate_random_test("branch-in-loop.template" "9" "branch-in-loop-9" "10-15")
generate_random_test("branch-in-loop.template" "10" "branch-in-loop-10" "10-15")

generate_random_test("loop-with-break.template" "1" "loop-with-break-1" "20-30")
generate_random_test("loop-with-break.template" "2" "loop-with-break-2" "20-30")
generate_random_test("loop-with-break.template" "3" "loop-with-break-3" "20-30")
generate_random_test("loop-with-break.template" "4" "loop-with-break-4" "20-30")
generate_random_test("loop-with-break.template" "5" "loop-with-break-5" "20-30")
generate_random_test("loop-with-break.template" "6" "loop-with-break-6" "20-30")
generate_random_test("loop-with-break.template" "7" "loop-with-break-7" "20-30")
generate_random_test("loop-with-break.template" "8" "loop-with-break-8" "20-30")
generate_random_test("loop-with-break.template" "9" "loop-with-break-9" "20-30")
generate_random_test("loop-with-break.template" "10" "loop-with-break-10" "20-30")
