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
  add_cudaq_library_install(${name})
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
  install(FILES ${name}.yml DESTINATION targets)
  configure_file(${name}.yml ${CMAKE_BINARY_DIR}/targets/${name}.yml COPYONLY)
endfunction()

function(add_target_mapping_arch providerName name)
  install(FILES ${name} DESTINATION targets/mapping/${providerName})
  configure_file(${name} ${CMAKE_BINARY_DIR}/targets/mapping/${providerName}/${name} COPYONLY)
endfunction()
