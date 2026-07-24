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
# MLIR libraries are linked as WHOLE_ARCHIVE so their full symbol set is
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

# 3. WHOLE_ARCHIVE the allowlisted MLIR libraries and C-API dependencies so their
# full symbol set is exported for downstream plugins.
cudaq_read_symbol_list(
  "${CMAKE_CURRENT_SOURCE_DIR}/mlir-libs-allowlist.txt" _cudaq_mlir_whole_archive)
get_property(_cudaq_required_mlir_libs GLOBAL PROPERTY CUDAQ_MLIR_REQUIRED_LIBS)
list(APPEND _cudaq_mlir_whole_archive ${_cudaq_required_mlir_libs})
list(REMOVE_DUPLICATES _cudaq_mlir_whole_archive)
foreach(_lib IN LISTS _cudaq_mlir_whole_archive)
  if(TARGET ${_lib})
    target_link_libraries(${LIBRARY_NAME} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_lib}>")
  else()
    message(WARNING
      "${LIBRARY_NAME}: MLIR library '${_lib}' not found; skipping whole-archive")
  endif()
endforeach()

# 4. Bundle the LLVM native target for JITing.
llvm_map_components_to_libnames(_cudaq_llvm_native_libs native nativecodegen)
foreach(_lib IN LISTS _cudaq_llvm_native_libs)
  if(TARGET ${_lib})
    target_link_libraries(${LIBRARY_NAME} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_lib}>")
  endif()
endforeach()

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

install(TARGETS ${LIBRARY_NAME}
        EXPORT cudaq-targets
        DESTINATION lib
        COMPONENT Runtime)
set_target_properties(${LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
