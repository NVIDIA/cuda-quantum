# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This file is derived from                                                    #
# https://github.com/llvm/circt/blob/main/cmake/modules/AddCIRCT.cmake         #
# CIRCT is an LLVM incubator project under Apache License 2.0 with LLVM        #
# Exceptions.                                                                  #
# ============================================================================ #

include_guard()

function(add_cudaq_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Dialect.td)
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(${dialect}DialectIncGen)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Ops.td)
  mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
  add_public_tablegen_target(${dialect}OpsIncGen)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Types.td)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  add_public_tablegen_target(${dialect}TypesIncGen)
  add_dependencies(cudaq-headers
    ${dialect}DialectIncGen ${dialect}OpsIncGen ${dialect}TypesIncGen)
endfunction()

function(add_cudaq_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(${interface}IncGen)
  add_dependencies(cudaq-headers ${interface}IncGen)
endfunction()

function(add_cudaq_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${CUDAQ_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
    OUTPUT ${GEN_DOC_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
    ${GEN_DOC_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(cudaq-doc ${output_id}DocGen)
endfunction()

function(add_cudaq_dialect_doc dialect dialect_namespace)
  add_cudaq_doc(${dialect} Dialects/${dialect} -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_cudaq_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL ENABLE_AGGREGATION)
endfunction()

# --------------------------------------------------------------------------- #
# add_cudaq_python_common_capi_library(<name> ...)``
#
# Drop-in replacement for MLIR's ``add_mlir_python_common_capi_library``
# that builds a **thin** common CAPI shared library (instead of statically
# linking the whole archive MLIR).
#
# Embeds only the CAPI object files themselves (``obj.<lib>`` from each
# ``EMBED_CAPI_LINK_LIBS`` entry); upstream MLIR and MLIRCAPI libs are stripped
# out and resolved dynamically from `cudaq::MLIR` resp. `cudaq::MLIRCAPI``.
#
# Accepts the same keyword arguments as MLIR's version:
#   ``INSTALL_COMPONENT``, ``INSTALL_DESTINATION``, ``OUTPUT_DIRECTORY``,
#   ``RELATIVE_INSTALL_ROOT``, ``DECLARED_HEADERS``, ``DECLARED_SOURCES``,
#   ``EMBED_LIBS``.
# --------------------------------------------------------------------------- #
function(add_cudaq_python_common_capi_library name)
  # 1. Parse arguments
  cmake_parse_arguments(ARG
    ""
    "INSTALL_COMPONENT;INSTALL_DESTINATION;OUTPUT_DIRECTORY;RELATIVE_INSTALL_ROOT"
    "DECLARED_HEADERS;DECLARED_SOURCES;EMBED_LIBS"
    ${ARGN})
  if(TARGET ${name})
    message(FATAL_ERROR "target ${name} already exists")
  endif()

  # 2. Collect object files from the C API libraries
  set(_embed_libs ${ARG_EMBED_LIBS})
  _flatten_mlir_python_targets(_all_source_targets ${ARG_DECLARED_SOURCES})
  foreach(_t ${_all_source_targets})
    get_target_property(_local_embed ${_t} mlir_python_EMBED_CAPI_LINK_LIBS)
    if(_local_embed)
      list(APPEND _embed_libs ${_local_embed})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _embed_libs)
  if(NOT _embed_libs)
    message(FATAL_ERROR "list of C API libraries cannot be empty")
  endif()

  set(_objects)
  foreach(_capi_lib IN LISTS _embed_libs)
    if(NOT TARGET obj.${_capi_lib})
      message(FATAL_ERROR "Ensure ${_capi_lib} was registered with ENABLE_AGGREGATION")
    endif()
    list(APPEND _objects "$<TARGET_OBJECTS:obj.${_capi_lib}>")
  endforeach()

  # 3. Create the shared library, with hidden visibility and linking to cudaqMLIR
  add_library(${name} SHARED ${_objects})
  target_link_libraries(${name} PRIVATE cudaq::cudaqMLIR)
  set_target_properties(${name} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
  if(ARG_OUTPUT_DIRECTORY)
    set_target_properties(${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
      RUNTIME_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
      ARCHIVE_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
      BINARY_OUTPUT_DIRECTORY  "${ARG_OUTPUT_DIRECTORY}")
  else()
    set_target_properties(${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  endif()

  # 4. Linker options: hide C++ symbols that would otherwise get re-exported from
  # cudaqMLIR. RPATH is configured below via mlir_python_setup_extension_rpath.
  if(APPLE)
    set(_exports "${CMAKE_CURRENT_BINARY_DIR}/${name}-exported.txt")
    file(WRITE "${_exports}" "_mlir*\n_cudaq*\n")
    set_property(TARGET ${name} APPEND PROPERTY LINK_DEPENDS "${_exports}")
    target_link_options(${name} PRIVATE
      "LINKER:-exported_symbols_list,${_exports}")
  else()
    set(_version_script "${CMAKE_CURRENT_BINARY_DIR}/${name}.map")
    file(WRITE "${_version_script}"
      "{\n  global:\n    mlir*;\n    cudaq*;\n  local:\n    *;\n};\n")
    set_property(TARGET ${name} APPEND PROPERTY
      LINK_DEPENDS "${_version_script}")
    target_link_options(${name} PRIVATE
      "LINKER:--version-script=${_version_script}")
  endif()

  # 5. RPATH (Python bindings): mlir_python_setup_extension_rpath sets
  # @loader_path / $ORIGIN; also append CUDAQ_LIBRARY_DIR for wheel layouts.
  mlir_python_setup_extension_rpath(${name}
    RELATIVE_INSTALL_ROOT "${ARG_RELATIVE_INSTALL_ROOT}")
  if(CUDAQ_LIBRARY_DIR)
    set_property(TARGET ${name} APPEND PROPERTY INSTALL_RPATH "${CUDAQ_LIBRARY_DIR}")
    set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH   "${CUDAQ_LIBRARY_DIR}")
  endif()

  # 6. Header sources target + install (add_mlir_python_common_capi_library parity)
  _flatten_mlir_python_targets(_flat_header_targets ${ARG_DECLARED_HEADERS})
  if(_flat_header_targets)
    set(_header_sources_target "${name}.sources")
    add_mlir_python_sources_target(${_header_sources_target}
      INSTALL_COMPONENT "${ARG_INSTALL_COMPONENT}"
      INSTALL_DIR       "${ARG_INSTALL_DESTINATION}/include"
      OUTPUT_DIRECTORY  "${ARG_OUTPUT_DIRECTORY}/include"
      SOURCES_TARGETS   ${_flat_header_targets})
    add_dependencies(${name} ${_header_sources_target})
  endif()
  if(ARG_INSTALL_COMPONENT AND ARG_INSTALL_DESTINATION)
    install(TARGETS ${name}
      COMPONENT "${ARG_INSTALL_COMPONENT}"
      LIBRARY   DESTINATION "${ARG_INSTALL_DESTINATION}"
      RUNTIME   DESTINATION "${ARG_INSTALL_DESTINATION}")
  endif()
endfunction()

# --------------------------------------------------------------------------- #
# ``add_cudaq_python_modules(<name> ...)``
#
# Drop-in wrapper around MLIR's ``add_mlir_python_modules``.  After the
# real assembly creates the ``<name>.extension.<module>.dso`` targets,
# this function:
#   - links ``cudaq::cudaqMLIR`` first (via ``target_link_options BEFORE``)
#     so MLIR/LLVM symbols resolve from the wheel dylib rather than from
#     static component archives embedded in the common CAPI lib.
#   - appends ``CUDAQ_LIBRARY_DIR`` to ``INSTALL_RPATH`` / ``BUILD_RPATH``
#     so the wheel's ``libcudaqMLIR.dylib`` resolves at load time.
# --------------------------------------------------------------------------- #
function(add_cudaq_python_modules name)
  # Delegate to MLIR's real implementation.
  add_mlir_python_modules(${name} ${ARGN})

  # Collect every *.extension.*.dso target created for this module set.
  get_property(_all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  list(FILTER _all_targets INCLUDE REGEX "^${name}\\.extension\\..*\\.dso$")

  foreach(_dso IN LISTS _all_targets)
    # Put cudaqMLIR BEFORE all other deps on the link line so its MLIR/LLVM
    # symbols shadow any static component archives in the common CAPI lib.
    target_link_libraries(${_dso} PRIVATE cudaq::cudaqMLIR)
    target_link_options(${_dso} BEFORE PRIVATE
      "$<TARGET_FILE:cudaq::cudaqMLIR>")

    if(CUDAQ_LIBRARY_DIR)
      set_property(TARGET ${_dso} APPEND PROPERTY INSTALL_RPATH "${CUDAQ_LIBRARY_DIR}")
      set_property(TARGET ${_dso} APPEND PROPERTY BUILD_RPATH   "${CUDAQ_LIBRARY_DIR}")
    endif()
  endforeach()
endfunction()

function(add_cudaq_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_DIALECT_LIBS ${name})
  add_cudaq_library(${ARGV} DEPENDS cudaq-headers)
endfunction()

function(add_cudaq_translation_library name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_TRANSLATION_LIBS ${name})
  add_cudaq_library(${ARGV} DEPENDS cudaq-headers)
endfunction()

function(add_target_config name)
  install(FILES ${name}.yml DESTINATION targets COMPONENT Runtime)
  configure_file(${name}.yml ${CMAKE_BINARY_DIR}/targets/${name}.yml COPYONLY)
endfunction()

function(add_target_mapping_arch providerName name)
  install(FILES ${name}
          DESTINATION targets/mapping/${providerName}
          COMPONENT Runtime)
  configure_file(${name} ${CMAKE_BINARY_DIR}/targets/mapping/${providerName}/${name} COPYONLY)
endfunction()

# Make `target` resolve its transitive CUDA-Q MLIR deps against static
# MLIR component libraries instead of libcudaqMLIR.so.
function(cudaq_use_static_mlir target)
  set_target_properties(${target} PROPERTIES CUDAQ_MLIR_STATIC ON)
endfunction()

# Build a shared MLIR extension against the CUDA-Q wheel layout.
#
# Usage (after find_package(CUDAQ REQUIRED) and include(AddCUDAQ)):
#   cudaq_add_mlir_extension(my_pass
#     SOURCES MyPass.cpp
#     LINK_LIBS cudaq::cudaq-mlir-runtime)
#
# Links libcudaqMLIR first so the extension shares the single MLIR/LLVM instance
# from the CUDA-Q wheel, and sets RPATH so CUDA-Q libraries resolve from the
# wheel's lib directory at load time.
function(cudaq_add_mlir_extension name)
  cmake_parse_arguments(ARG "" "DESTINATION" "SOURCES;LINK_LIBS" ${ARGN})

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "cudaq_add_mlir_extension(${name}): SOURCES is required")
  endif()

  add_library(${name} SHARED ${ARG_SOURCES})

  # cudaqMLIR must come first so downstream shares its MLIR/LLVM instance.
  target_link_libraries(${name} PRIVATE cudaq::cudaqMLIR ${ARG_LINK_LIBS})

  if(APPLE)
    set_property(TARGET ${name} APPEND PROPERTY INSTALL_RPATH "@loader_path")
  else()
    set_property(TARGET ${name} APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
  endif()

  if(CUDAQ_LIBRARY_DIR)
    set_property(TARGET ${name} APPEND PROPERTY INSTALL_RPATH "${CUDAQ_LIBRARY_DIR}")
    set_property(TARGET ${name} PROPERTY BUILD_RPATH "${CUDAQ_LIBRARY_DIR}")
  endif()

  if(ARG_DESTINATION)
    install(TARGETS ${name} DESTINATION ${ARG_DESTINATION})
  endif()
endfunction()

# Provide canonical CMake imported interface targets for core CUDA-Q MLIR libraries
add_library(cudaq::MLIR INTERFACE IMPORTED)
set_target_properties(cudaq::MLIR PROPERTIES
  INTERFACE_LINK_LIBRARIES cudaq::cudaqMLIR
)
