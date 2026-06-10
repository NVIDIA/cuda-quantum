# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_custom_target(CudaQUnitTests)
# The following property will be used to keep track of the unit test targets 
# that we create with the `add_cudaq_unittest` function. We will use this 
# property to add the tests to the `CudaQUnitTests` target and to create custom 
# targets for each test that we can run with 
# `{make,ninja, cmake --build build -target} check-<test_label>`.
set_property(GLOBAL PROPERTY CUDAQ_UNIT_TEST_TARGETS "")


function(create_cudaq_unittest_check_target test_name check_target_name)
    add_custom_target(check-${check_target_name}
        COMMAND $<TARGET_FILE:${test_name}>
        DEPENDS ${test_name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
    )
endfunction(create_cudaq_unittest_check_target)

# This is a helper function to add a unit test to the CudaQUnitTests target. 
# It creates an executable for the test and it takes the test name and a test label 
# as arguments, as well as optional arguments for:
# - sources
# - includes
# - linked libraries
# - compile options
# - compile definitions 
# for the test. 
# The test will be added to the `CudaQUnitTests` target, and a custom target will 
# be created for it that we can run with `make check-<test_label>`.
function(add_cudaq_unittest test_name test_label)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs
        SOURCES
        INCLUDES
        LINK_OPTS
        LINKED_LIBS
        COMPILE_OPTS
        COMPILE_DEFS
    )


    cmake_parse_arguments(USER_ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${test_name}
        ${USER_ARG_SOURCES}
    )

    if (USER_ARG_INCLUDES)
        target_include_directories(${test_name}
            PRIVATE
                ${USER_ARG_INCLUDES}
        )
    endif()

    if (USER_ARG_LINKED_LIBS)
        target_link_libraries(${test_name}
            PRIVATE
                ${USER_ARG_LINKED_LIBS}
                gtest_main
        )
    endif()

    if (USER_ARG_LINK_OPTS)
        target_link_options(${test_name}
            PRIVATE
                ${USER_ARG_LINK_OPTS}
        )
    endif()


    if (USER_ARG_COMPILE_DEFS)
        target_compile_definitions(${test_name}
            PRIVATE
                ${USER_ARG_COMPILE_DEFS}
        )
    endif()

    if (USER_ARG_COMPILE_OPTS)
        target_compile_options(${test_name}
            PRIVATE
                ${USER_ARG_COMPILE_OPTS}
        )
    endif()

    add_dependencies(CudaQUnitTests ${test_name})

    set_property(GLOBAL APPEND PROPERTY CUDAQ_UNIT_TEST_TARGETS ${test_name})
endfunction(add_cudaq_unittest)
