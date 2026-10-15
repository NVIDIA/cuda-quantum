# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Target-database generation for the CUDAQ target component.

# Generates the precompiled target database from every .yml under
# cudaq/lib/Targets/ plus any out-of-tree add_target_config() registrations.
# Source paths resolve relative to this module (at call time, so the function
# works when called from the top-level CMakeLists.txt scope as well as from
# the standalone component build).
function(cudaq_finalize_target_database)
  get_filename_component(CUDAQ_TARGET_DB_REPO_ROOT
                         "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../.."
                         ABSOLUTE)
  file(GLOB _target_dirs LIST_DIRECTORIES true
    ${CUDAQ_TARGET_DB_REPO_ROOT}/cudaq/lib/Targets/*)
  set(_gen_args)
  set(_deps)
  set(_count 0)
  set(_seen_names)
  foreach(_dir ${_target_dirs})
    if (NOT IS_DIRECTORY ${_dir})
      continue()
    endif()
    get_filename_component(_name ${_dir} NAME)
    set(_yml ${_dir}/${_name}.yml)
    if (NOT EXISTS ${_yml})
      continue()
    endif()
    list(APPEND _gen_args "${_name}=${_yml}")
    list(APPEND _deps ${_yml})
    list(APPEND _seen_names ${_name})
    math(EXPR _count "${_count} + 1")
  endforeach()

  get_property(_ext_names GLOBAL PROPERTY CUDAQ_TARGET_DB_NAMES)
  get_property(_ext_paths GLOBAL PROPERTY CUDAQ_TARGET_DB_PATHS)
  list(LENGTH _ext_names _ext_count)
  if (_ext_count GREATER 0)
    math(EXPR _ext_last_idx "${_ext_count} - 1")
    foreach(_idx RANGE ${_ext_last_idx})
      list(GET _ext_names ${_idx} _name)
      list(GET _ext_paths ${_idx} _path)
      if (_name IN_LIST _seen_names)
        continue()
      endif()
      string(FIND "${_path}" "${CUDAQ_TARGET_DB_REPO_ROOT}/cudaq/lib/Targets/" _idx_pos)
      if (_idx_pos EQUAL 0)
        continue() # already covered by the static glob above
      endif()
      list(APPEND _gen_args "${_name}=${_path}")
      list(APPEND _deps ${_path})
      list(APPEND _seen_names ${_name})
      math(EXPR _count "${_count} + 1")
    endforeach()
  endif()

  if (_count EQUAL 0)
    message(FATAL_ERROR
      "cudaq_finalize_target_database: no target YAMLs found under "
      "${CUDAQ_TARGET_DB_REPO_ROOT}/cudaq/lib/Targets")
  endif()

  set(_gen_cpp ${CMAKE_BINARY_DIR}/generated/TargetDatabase.gen.cpp)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)
  add_custom_command(
    OUTPUT ${_gen_cpp}
    COMMAND $<TARGET_FILE:cudaq-target-db-gen> -o ${_gen_cpp} ${_gen_args}
    DEPENDS cudaq-target-db-gen ${_deps}
    COMMENT "Generating precompiled cudaq/ target database (${_count} targets)"
    VERBATIM)

  # Mark dependency on the generated source file.
  add_custom_target(CUDAQTargetDatabaseGen DEPENDS ${_gen_cpp})
  add_dependencies(CUDAQTargetDatabase CUDAQTargetDatabaseGen)
  target_sources(CUDAQTargetDatabase PRIVATE ${_gen_cpp})
endfunction()
