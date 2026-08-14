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
# It bundles every library in CUDAQ_MLIR_BUNDLED_LIBS:
#  - CUDA-Q dialect/transform libs (obj.<lib> targets) as object files
#  - MLIR/LLVM libs as WHOLE_ARCHIVE static dependencies
#
# MLIR libraries are linked as WHOLE_ARCHIVE so their full symbol set is
# exported for downstream plugins.
################################################################################

set(LIBRARY_NAME cudaqMLIR)

# 1. Partition each library in CUDAQ_MLIR_BUNDLED_LIBS as either to be:
#    - bundled directly as object files (preferred if available), or
#    - whole-archived as a static lib otherwise.
set(_cudaq_bundle_objs)
set(_cudaq_bundle_whole_archive_libs)
set(_cudaq_bundle_link_libs)
foreach(_lib IN LISTS CUDAQ_MLIR_BUNDLED_LIBS)
  if(TARGET obj.${_lib})
    list(APPEND _cudaq_bundle_objs "$<TARGET_OBJECTS:obj.${_lib}>")
    list(APPEND _cudaq_bundle_link_libs ${_lib})
    # Do not export inline/template member functions, as downstream re-emits
    # them from headers anyway.
    target_compile_options(obj.${_lib} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>"
      -ffunction-sections -fdata-sections)
  elseif(TARGET ${_lib})
    list(APPEND _cudaq_bundle_whole_archive_libs ${_lib})
  else()
    message(WARNING
      "${LIBRARY_NAME}: bundled library '${_lib}' not found; skipping")
  endif()
endforeach()

# 2a. Create the shared library from the list of object files.
add_library(${LIBRARY_NAME} SHARED ${_cudaq_bundle_objs})

# 2b. Provide the namespaced alias cudaq::cudaqMLIR used downstream
add_library(cudaq::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

# 2c. Pull in any required transitive dependencies.
target_link_libraries(${LIBRARY_NAME} PRIVATE ${_cudaq_bundle_link_libs})

# 3. WHOLE_ARCHIVE MLIR/LLVM static libs so their full symbol set is exported
# for downstream plugins.
foreach(_lib IN LISTS _cudaq_bundle_whole_archive_libs)
  target_link_libraries(${LIBRARY_NAME} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_lib}>")
endforeach()

# Ideally, we use -Bsymbolic-functions as it removes PLT/GOT indirection. However
# that also opts the library out of ELF symbol interposition, which we currently
# rely on to deduplicate symbols when libstdc++/libgcc are statically linked into
# multiple CUDA-Q shared objects (otherwise exception unwinding breaks).
if(NOT APPLE AND NOT CUDAQ_STATIC_CXX_RUNTIME)
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
