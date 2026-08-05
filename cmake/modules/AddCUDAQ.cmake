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

# Define `CUDAQ_MLIR_BUNDLED_LIBS_PATH`: the file that lists all bundled MLIR libraries. In-tree
# it lives in the source tree; installed it sits next to this file in lib/cmake/cudaq.
foreach(_candidate
  "${CMAKE_CURRENT_LIST_DIR}/../../cudaq/lib/Optimizer/mlir-bundled-libs.txt"
  "${CMAKE_CURRENT_LIST_DIR}/mlir-bundled-libs.txt")
  if(EXISTS "${_candidate}")
    get_filename_component(CUDAQ_MLIR_BUNDLED_LIBS_PATH "${_candidate}" ABSOLUTE)
    break()
  endif()
endforeach()
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
if(CUDAQ_MLIR_BUNDLED_LIBS_PATH)
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

# Build a thin shared C API library.
#
# The listed C API libraries are embedded via their object targets without
# inheriting their static MLIR link interfaces. Their C++ dependencies are
# recorded in CUDAQ_MLIR_REQUIRED_LIBS for the single cudaqMLIR DSO to provide.
function(add_cudaq_capi_shared_library name)
  # 1. Parse arguments
  if(NOT ARGN)
    message(FATAL_ERROR "list of C API libraries cannot be empty")
  endif()
  if(TARGET ${name})
    message(FATAL_ERROR "target ${name} already exists")
  endif()

  # 2. Collect object files from the C API libraries
  set(_objects)
  foreach(_capi_lib IN LISTS ARGN)
    if(NOT TARGET obj.${_capi_lib})
      message(FATAL_ERROR "Ensure ${_capi_lib} was registered with ENABLE_AGGREGATION")
    endif()
    list(APPEND _objects "$<TARGET_OBJECTS:obj.${_capi_lib}>")

    # 3. Record MLIR dependencies of the C API libraries (to be whole-archived into cudaqMLIR)
    get_target_property(_capi_deps ${_capi_lib}
      MLIR_AGGREGATE_DEP_LIBS_IMPORTED)
    foreach(_dep IN LISTS _capi_deps)
      if(TARGET ${_dep}
          AND NOT _dep IN_LIST ARGN
          AND NOT _dep MATCHES "CAPI"
          AND NOT _dep STREQUAL "cudaqMLIR")
        set_property(GLOBAL APPEND PROPERTY CUDAQ_MLIR_REQUIRED_LIBS "${_dep}")
      endif()
    endforeach()
  endforeach()

  # 4. Create the shared library, with hidden visibility and linking to cudaqMLIR
  add_library(${name} SHARED ${_objects})
  target_link_libraries(${name} PRIVATE cudaqMLIR)
  set_target_properties(${name} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

  # 5. Linker options: set RPATH and hide all C++ symbols that would otherwise get
  # re-exported from cudaqMLIR
  if(APPLE)
    set_property(TARGET ${name} PROPERTY INSTALL_RPATH "@loader_path")
    set(_exports "${CMAKE_CURRENT_BINARY_DIR}/${name}-exported.txt")
    file(WRITE "${_exports}" "_mlir*\n_cudaq*\n")
    set_property(TARGET ${name} APPEND PROPERTY LINK_DEPENDS "${_exports}")
    target_link_options(${name} PRIVATE
      "LINKER:-exported_symbols_list,${_exports}")
  else()
    set_property(TARGET ${name} PROPERTY INSTALL_RPATH "$ORIGIN")
    set(_version_script "${CMAKE_CURRENT_BINARY_DIR}/${name}.map")
    file(WRITE "${_version_script}"
      "{\n  global:\n    mlir*;\n    cudaq*;\n  local:\n    *;\n};\n")
    set_property(TARGET ${name} APPEND PROPERTY
      LINK_DEPENDS "${_version_script}")
    target_link_options(${name} PRIVATE
      "LINKER:--version-script=${_version_script}")
  endif()

  # Check for unexpected undefined symbols
  cudaq_check_mlir_symbol_closure(${name})
endfunction()

# Adds a CUDA Quantum dialect library target for installation. This should normally
# only be called from add_cudaq_dialect_library().
function(add_cudaq_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT CUDAQTargets)
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
