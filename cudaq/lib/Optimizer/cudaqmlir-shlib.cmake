# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

################################################################################
# Define the shared core library libcudaqMLIR.so
#
# It bundles the following compilation units into one shared library:
#  - All the object files of the libraries registered via register_cudaq_mlir_lib
#  - All the MLIR libraries listed in mlir-libs-allowlist.txt
#
# We mark the MLIR libraries as WHOLE_ARCHIVE dependencies so their full symbol set is
# exported for downstream plugins.
################################################################################

# Read a newline-separated list file (one entry per line) into `_out_var`,
# stripping comments and whitespace.
function(cudaq_read_symbol_list _file _out_var)
  # Re-run CMake configuration if the file changes.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
  file(STRINGS "${_file}" _lines)
  set(_entries)
  foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line)
    if(NOT (_line STREQUAL "" OR _line MATCHES "^#"))
      list(APPEND _entries "${_line}")
    endif()
  endforeach()
  set(${_out_var} "${_entries}" PARENT_SCOPE)
endfunction()

set(LIBRARY_NAME cudaqMLIR)
get_property(_cudaq_bundle_libs GLOBAL PROPERTY CUDAQ_MLIR_BUNDLE_LIBS)

# 1. Bundle the list of all MLIR object files. This assumes that every bundled
# lib was built with ENABLE_AGGREGATION (e.g. via add_cudaq_library)
set(_cudaq_bundle_objs)
foreach(_lib IN LISTS _cudaq_bundle_libs)
  if(TARGET obj.${_lib})
    list(APPEND _cudaq_bundle_objs "$<TARGET_OBJECTS:obj.${_lib}>")
    # Do not export inline/template member functions, as downstream re-emits
    # them from headers anyway.
    target_compile_options(obj.${_lib} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>"
      -ffunction-sections -fdata-sections)
  else()
    message(WARNING
      "${LIBRARY_NAME}: obj.${_lib} not found; ensure ${_lib} is built with ENABLE_AGGREGATION")
  endif()
endforeach()
add_library(${LIBRARY_NAME} SHARED ${_cudaq_bundle_objs})

# 2. Pull in the dependencies
target_link_libraries(${LIBRARY_NAME} PRIVATE ${_cudaq_bundle_libs})

# 3. WHOLE_ARCHIVE the allowlisted MLIR libraries so their full symbol set is
# exported for downstream plugins.
cudaq_read_symbol_list(
  "${CMAKE_CURRENT_SOURCE_DIR}/mlir-libs-allowlist.txt" _cudaq_mlir_whole_archive)
foreach(_lib IN LISTS _cudaq_mlir_whole_archive)
  if(TARGET ${_lib})
    target_link_libraries(${LIBRARY_NAME} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_lib}>")
  else()
    message(WARNING
      "${LIBRARY_NAME}: MLIR library '${_lib}' not found; skipping whole-archive")
  endif()
endforeach()

# 4. Hide all symbols from the blocklist.
cudaq_read_symbol_list(
  "${CMAKE_CURRENT_SOURCE_DIR}/mlir-symbols-blocklist.txt" _cudaq_blocklist_patterns)
if(APPLE)
  # ld64: one glob per line; localize via -unexported_symbols_list.
  set(_cudaq_symbol_list "${CMAKE_CURRENT_BINARY_DIR}/cudaqMLIR-unexported.txt")
  # relink if the symbol list changes
  set_property(TARGET ${LIBRARY_NAME} APPEND PROPERTY LINK_DEPENDS "${_cudaq_symbol_list}")
  string(REPLACE ";" "\n" _cudaq_unexported "${_cudaq_blocklist_patterns}")
  file(WRITE "${_cudaq_symbol_list}" "${_cudaq_unexported}\n")
  target_link_options(${LIBRARY_NAME} PRIVATE
    "LINKER:-unexported_symbols_list,${_cudaq_symbol_list}")
else()
  # GNU ld / lld: a version script with a local: block leaves the rest exported.
  set(_cudaq_version_script "${CMAKE_CURRENT_BINARY_DIR}/cudaqMLIR-hidden.map")
  # relink if the version script changes
  set_property(TARGET ${LIBRARY_NAME} APPEND PROPERTY LINK_DEPENDS "${_cudaq_version_script}")
  set(_cudaq_blocklist_body "")
  foreach(_pat IN LISTS _cudaq_blocklist_patterns)
    string(APPEND _cudaq_blocklist_body "    ${_pat};\n")
  endforeach()
  file(WRITE "${_cudaq_version_script}" "{\n  local:\n${_cudaq_blocklist_body}};\n")
  target_link_options(${LIBRARY_NAME} PRIVATE
    "LINKER:--version-script=${_cudaq_version_script}")
endif()

# We are using the following linker flags:
#   - -Bsymbolic-functions: ELF treats symbols as preemptible by default. This adds a
#      PLT/GOT indirection overhead that we remove.
#   - -dead_strip/--gc-sections: Garbage-collect functions/data not reachable from
#      exported symbols.
if(NOT APPLE)
  target_link_options(${LIBRARY_NAME} PRIVATE "LINKER:-Bsymbolic-functions")
endif()

# Garbage-collect functions/data not reachable from exported symbols.
if(APPLE)
  target_link_options(${LIBRARY_NAME} PRIVATE "LINKER:-dead_strip")
else()
  target_link_options(${LIBRARY_NAME} PRIVATE "LINKER:--gc-sections")
endif()

install(TARGETS ${LIBRARY_NAME} EXPORT CUDAQTargets DESTINATION lib)
set_target_properties(${LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
