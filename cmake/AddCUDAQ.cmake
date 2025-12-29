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

# Set up MLIRExecutionEngine for use with libMLIR.dylib.
# Uses MLIRExecutionEngineShared (backported from LLVM 19 for LLVM 16).
# TODO: Remove this function when upgrading to LLVM 19+.
function(cudaq_setup_mlir_execution_engine_dylib)
  if(TARGET MLIRExecutionEngineShared AND NOT TARGET CUDAQMLIRExecutionEngine)
    add_library(CUDAQMLIRExecutionEngine INTERFACE)
    target_link_libraries(CUDAQMLIRExecutionEngine INTERFACE MLIRExecutionEngineShared)
  endif()
endfunction()

# Get libraries with MLIR dylib substitution when available.
# When libMLIR.dylib exists (built with MLIR_LINK_MLIR_DYLIB=ON), replaces all
# MLIR* libraries with just "MLIR". Non-MLIR libraries pass through unchanged.
# MLIRExecutionEngine is special-cased to use CUDAQMLIRExecutionEngine wrapper.
# TODO: Replace with mlir_target_link_libraries() when upgrading to LLVM 17+.
# See: https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIR.cmake
function(cudaq_get_mlir_libs output_var)
  if(TARGET MLIR)
    # Ensure ExecutionEngine wrapper exists
    cudaq_setup_mlir_execution_engine_dylib()
    set(result)
    set(added_mlir FALSE)
    set(added_exec_engine FALSE)
    foreach(lib ${ARGN})
      if(lib STREQUAL "MLIRExecutionEngine" OR lib STREQUAL "MLIRExecutionEngineUtils")
        # Use our wrapper that redirects MLIR deps to dylib
        if(NOT added_exec_engine)
          list(APPEND result CUDAQMLIRExecutionEngine)
          set(added_exec_engine TRUE)
        endif()
      elseif(lib MATCHES "^MLIR" AND NOT lib STREQUAL "MLIR")
        # Replace MLIR static libs with the dylib (add once)
        if(NOT added_mlir)
          list(APPEND result MLIR)
          set(added_mlir TRUE)
        endif()
      else()
        list(APPEND result ${lib})
      endif()
    endforeach()
    set(${output_var} ${result} PARENT_SCOPE)
  else()
    set(${output_var} ${ARGN} PARENT_SCOPE)
  endif()
endfunction()

function(add_cudaq_library name)
  cudaq_get_mlir_libs(new_args ${ARGN})
  add_mlir_library(${name} ${new_args} DISABLE_INSTALL)
  add_cudaq_library_install(${name})
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
