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
  set(PROCESS_DOC_SCRIPT
    ${CUDAQ_SOURCE_DIR}/cmake/modules/ProcessMLIRMarkdown.cmake)
  add_custom_command(
    OUTPUT ${GEN_DOC_FILE}
    COMMAND ${CMAKE_COMMAND}
    -DINPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
    -DOUTPUT_FILE=${GEN_DOC_FILE}
    -P ${PROCESS_DOC_SCRIPT}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md ${PROCESS_DOC_SCRIPT}
    VERBATIM)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(cudaq-doc ${output_id}DocGen)
endfunction()

function(add_cudaq_dialect_doc dialect dialect_namespace)
  add_cudaq_doc(${dialect}Ops Dialects/${dialect}
    -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_cudaq_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL ENABLE_AGGREGATION)
  add_cudaq_library_install(${name})
endfunction()

# Define `CUDAQ_MLIR_BUNDLED_LIBS_PATH`: the file that lists all bundled MLIR libraries.
# In-tree and installed it sits next to this file (lib/cmake/cudaq when installed).
set(CUDAQ_MLIR_BUNDLED_LIBS_PATH "${CMAKE_CURRENT_LIST_DIR}/mlir-bundled-libs.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CUDAQ_MLIR_BUNDLED_LIBS_PATH}")

# Read a newline-separated list file (one entry per line) into ``<out_var>``,
# stripping comments and whitespace.
function(_cudaq_read_symbol_list _file _out_var)
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

# Define `CUDAQ_MLIR_BUNDLED_LIBS`: the single list of MLIR libraries bundled into
# libcudaqMLIR.
if(EXISTS "${CUDAQ_MLIR_BUNDLED_LIBS_PATH}")
  _cudaq_read_symbol_list("${CUDAQ_MLIR_BUNDLED_LIBS_PATH}" CUDAQ_MLIR_BUNDLED_LIBS)
  # Target-specific codegen is not in the list because it differs per
  # architecture: LLVMX86* on x86_64, LLVMAArch64* on arm64.
  if(COMMAND llvm_map_components_to_libnames)
    llvm_map_components_to_libnames(_cudaq_llvm_native_libs native nativecodegen)
    list(APPEND CUDAQ_MLIR_BUNDLED_LIBS ${_cudaq_llvm_native_libs})
  endif()
  list(REMOVE_DUPLICATES CUDAQ_MLIR_BUNDLED_LIBS)
endif()

# --------------------------------------------------------------------------- #
# ``cudaq_check_mlir_symbol_closure(<target>)``
#
# Fail the build if ``<target>`` references MLIR/LLVM symbols that libcudaqMLIR
# does not export. Everything in CUDA-Q must resolve MLIR/LLVM dynamically from the
# single libcudaqMLIR instance. See scripts/check_mlir_symbols.sh.
# --------------------------------------------------------------------------- #

option(CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE
  "Fail the build when a library references MLIR/LLVM symbols libcudaqMLIR does not export."
  ON)

# In-tree this module sits in cmake/modules; installed it sits in
# lib/cmake/cudaq next to a copy of the script.
foreach(_candidate
  "${CMAKE_CURRENT_LIST_DIR}/../../scripts/check_mlir_symbols.sh"
  "${CMAKE_CURRENT_LIST_DIR}/check_mlir_symbols.sh")
  if(EXISTS "${_candidate}")
    get_filename_component(CUDAQ_CHECK_SYMBOL_SCRIPT "${_candidate}" ABSOLUTE)
    break()
  endif()
endforeach()

function(cudaq_check_mlir_symbol_closure name)
  if(NOT CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE OR NOT CUDAQ_CHECK_SYMBOL_SCRIPT)
    return()
  endif()
  add_custom_command(TARGET ${name} POST_BUILD
    COMMAND bash "${CUDAQ_CHECK_SYMBOL_SCRIPT}"
    "$<TARGET_FILE:${name}>" "$<TARGET_FILE:cudaq::cudaqMLIR>"
    COMMENT "Checking MLIR/LLVM symbol closure of ${name}"
    VERBATIM)
endfunction()

# --------------------------------------------------------------------------- #
# add_cudaq_python_common_capi_library(<name> ...)``
#
# Drop-in replacement for MLIR's ``add_mlir_python_common_capi_library``
# that builds a common CAPI shared library without duplicating upstream MLIR.
#
# Identical to upstream except that the static MLIR/LLVM archives already
# contained in ``libcudaqMLIR`` are excluded from the aggregate and resolved
# dynamically from it instead, so the C API library holds no second copy of
# MLIR. Project-owned dependencies (e.g. a downstream project's dialect
# libraries) are still linked in.
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

  # C-APIs are required to be defined with ENABLE_AGGREGATION on (on by default).
  foreach(_capi_lib IN LISTS _embed_libs)
    if(NOT TARGET obj.${_capi_lib})
      message(FATAL_ERROR "Ensure ${_capi_lib} was registered with ENABLE_AGGREGATION")
    endif()
  endforeach()

  # 3. Create the shared library, with hidden visibility and linking to cudaqMLIR
  #
  # We use the MLIR-provided aggregation utility but modify it to exclude any
  # libraries provided by `libcudaqMLIR.so` and instead link in `cudaq::cudaqMLIR`.
  # We then hide all symbols by default (same as add_mlir_python_common_capi_library).
  add_mlir_aggregate(${name}
    SHARED
    DISABLE_INSTALL
    EMBED_LIBS ${_embed_libs}
    PUBLIC_LIBS cudaq::cudaqMLIR)

  set_property(TARGET ${name} APPEND PROPERTY
    MLIR_AGGREGATE_EXCLUDE_LIBS ${CUDAQ_MLIR_BUNDLED_LIBS})

  set_target_properties(${name} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
  if(ARG_OUTPUT_DIRECTORY)
    set_target_properties(${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
      RUNTIME_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
      ARCHIVE_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
      BINARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}")
  else()
    set_target_properties(${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  endif()

  # Check for unexpected undefined symbols
  cudaq_check_mlir_symbol_closure(${name})

  # 4. RPATH (Python bindings): mlir_python_setup_extension_rpath sets
  # @loader_path / $ORIGIN; also append CUDAQ_LIBRARY_DIR for wheel layouts.
  mlir_python_setup_extension_rpath(${name}
    RELATIVE_INSTALL_ROOT "${ARG_RELATIVE_INSTALL_ROOT}")
  if(CUDAQ_LIBRARY_DIR)
    set_property(TARGET ${name} APPEND PROPERTY INSTALL_RPATH "${CUDAQ_LIBRARY_DIR}")
    set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "${CUDAQ_LIBRARY_DIR}")
  endif()

  # 5. Header sources target + install (copied from add_mlir_python_common_capi_library)
  _flatten_mlir_python_targets(_flat_header_targets ${ARG_DECLARED_HEADERS})
  if(_flat_header_targets)
    set(_header_sources_target "${name}.sources")
    add_mlir_python_sources_target(${_header_sources_target}
      INSTALL_COMPONENT "${ARG_INSTALL_COMPONENT}"
      INSTALL_DIR "${ARG_INSTALL_DESTINATION}/include"
      OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}/include"
      SOURCES_TARGETS ${_flat_header_targets})
    add_dependencies(${name} ${_header_sources_target})
  endif()
  if(ARG_INSTALL_COMPONENT AND ARG_INSTALL_DESTINATION)
    install(TARGETS ${name}
      COMPONENT "${ARG_INSTALL_COMPONENT}"
      LIBRARY DESTINATION "${ARG_INSTALL_DESTINATION}"
      RUNTIME DESTINATION "${ARG_INSTALL_DESTINATION}")
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

  # Fix RPATH for wheel layout. Always use relative paths for wheel delocation.
  if(APPLE)
    set(_origin_prefix "@loader_path")
  else()
    set(_origin_prefix "$ORIGIN")
  endif()
  if(SKBUILD)
    set(_cudaq_python_install_rpaths
      "${_origin_prefix}/../../../lib"
      "${_origin_prefix}/../../../cuda_quantum.libs")
  else()
    set(_cudaq_python_install_rpaths
      "${_origin_prefix}/../../../lib"
      "${_origin_prefix}/../../../lib/plugins")
  endif()

  # Collect every *.extension.*.dso target created for this module set.
  get_property(_all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  list(FILTER _all_targets INCLUDE REGEX "^${name}\\.extension\\.")

  foreach(_dso IN LISTS _all_targets)
    # Put cudaqMLIR BEFORE all other deps on the link line so its MLIR/LLVM
    # symbols shadow any static component archives in the common CAPI lib.
    target_link_libraries(${_dso} PRIVATE cudaq::cudaqMLIR)
    target_link_options(${_dso} BEFORE PRIVATE
      "$<TARGET_FILE:cudaq::cudaqMLIR>")

    set_property(TARGET ${_dso} APPEND PROPERTY
      INSTALL_RPATH ${_cudaq_python_install_rpaths})
    # BUILD_RPATH may use the absolute build lib dir so the bindings can run
    # from the build tree (e.g. ctest-driven python tests).
    if(CUDAQ_LIBRARY_DIR)
      set_property(TARGET ${_dso} APPEND PROPERTY BUILD_RPATH "${CUDAQ_LIBRARY_DIR}")
    endif()

    cudaq_check_mlir_symbol_closure(${_dso})
  endforeach()
endfunction()

# Adds a CUDA Quantum dialect library target for installation. This should normally
# only be called from add_cudaq_library().
function(add_cudaq_library_install name)
  install(TARGETS ${name} COMPONENT Development EXPORT CUDAQTargets)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY CUDAQ_EXPORTS ${name})
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

# Define the public alias ``cudaq::MLIR`` for use in downstream projects.
if(NOT TARGET cudaq::MLIR)
  add_library(cudaq::MLIR INTERFACE IMPORTED GLOBAL)
  set_target_properties(cudaq::MLIR PROPERTIES
    INTERFACE_LINK_LIBRARIES cudaq::cudaqMLIR
  )
endif()
