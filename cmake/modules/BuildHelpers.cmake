# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard()

function(_cudaq_check_openmp_usable RESULT_VAR)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    include(CheckCXXCompilerFlag)
    set(CMAKE_REQUIRED_FLAGS "${OpenMP_CXX_FLAGS}")
    check_cxx_compiler_flag("${OpenMP_CXX_FLAGS}" CUDAQ_HAS_OPENMP_FLAG)
    unset(CMAKE_REQUIRED_FLAGS)
    set(${RESULT_VAR} ${CUDAQ_HAS_OPENMP_FLAG} PARENT_SCOPE)
endfunction()

# If OpenMP is enabled and found, adds the necessary compile definitions to the
# given target, and the necessary dependencies to the given list of dependencies.
function(add_openmp_configurations TARGET_NAME DEPENDENCIES)
    _cudaq_check_openmp_usable(_openmp_usable)
    if(_openmp_usable)
        message(STATUS "OpenMP Found. Adding build flags to target ${TARGET_NAME}: ${OpenMP_CXX_FLAGS}.")
        list(APPEND ${DEPENDENCIES} OpenMP::OpenMP_CXX)
        set(${DEPENDENCIES} "${${DEPENDENCIES}}" PARENT_SCOPE)
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_OPENMP)
    elseif (CUDAQ_REQUIRE_OPENMP)
        message(FATAL_ERROR "OpenMP not found or compiler rejects OpenMP flags.")
    endif()
endfunction()

# If OpenMP is enabled and found, adds the necessary compile definitions to the
# interface dependencies of the given target.
function(add_openmp_interface_definitions TARGET_NAME)
    _cudaq_check_openmp_usable(_openmp_usable)
    if(_openmp_usable)
        message(STATUS "OpenMP Found. Adding interface definitions to target ${TARGET_NAME}.")
        target_compile_definitions(${TARGET_NAME} INTERFACE HAS_OPENMP)
    elseif (CUDAQ_REQUIRE_OPENMP)
        message(FATAL_ERROR "OpenMP not found or compiler rejects OpenMP flags.")
    endif()
endfunction()

# Making a NVQIR backend lib or config file available inside wheel
function(add_target_libs_to_wheel nvqir_backend_lib_or_config)
    if (NOT EXISTS "${nvqir_backend_lib_or_config}")
        message(FATAL_ERROR "Invalid file path to NVQIR backend lib or config: ${nvqir_backend_lib_or_config}.")
    endif()
    get_filename_component(FILE_EXTENSION ${nvqir_backend_lib_or_config} EXT)
    if ("${FILE_EXTENSION}" STREQUAL ".so")
        message(STATUS "Installing NVQIR backend lib '${nvqir_backend_lib_or_config}'")
        install(FILES ${nvqir_backend_lib_or_config} DESTINATION lib)
    elseif("${FILE_EXTENSION}" STREQUAL ".yml")
        message(STATUS "Installing NVQIR backend config '${nvqir_backend_lib_or_config}'")
        install(FILES ${nvqir_backend_lib_or_config} DESTINATION targets)
    else()
        message(WARNING "Unknown file extension of ${nvqir_backend_lib_or_config} file. It will be ignored.")
    endif()
endfunction()

# Stage python sources into the build tree, one symlink rule per file.
# Modeled on MLIR's add_mlir_python_sources_target minus its install and
# export machinery; these sources are installed through other mechanisms.
function(cudaq_stage_python_sources name)
    cmake_parse_arguments(ARG "" "ROOT_DIR;OUTPUT_DIRECTORY" "SOURCES" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unhandled arguments to cudaq_stage_python_sources(${name}): ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    set(_dest_paths "")
    foreach(_rel_path ${ARG_SOURCES})
        set(_src_path "${ARG_ROOT_DIR}/${_rel_path}")
        set(_dest_path "${ARG_OUTPUT_DIRECTORY}/${_rel_path}")
        get_filename_component(_dest_dir "${_dest_path}" DIRECTORY)
        file(MAKE_DIRECTORY "${_dest_dir}")
        add_custom_command(
            OUTPUT "${_dest_path}"
            COMMENT "Staging python source ${_rel_path}"
            DEPENDS "${_src_path}"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${_src_path}" "${_dest_path}"
        )
        list(APPEND _dest_paths "${_dest_path}")
    endforeach()

    add_custom_target(${name} DEPENDS ${_dest_paths})
endfunction()
