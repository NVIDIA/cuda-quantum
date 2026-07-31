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
endfunction()

# Read a newline-separated list file (one entry per line) into ``_out_var``,
# stripping comments and whitespace.
function(cudaq_read_symbol_list _file _out_var)
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

# MLIR/LLVM libraries already provided by ``libcudaqMLIR.so``.
#
# cudaqMLIR-shlib.cmake writes the resolved list to cudaqMLIR-contents.txt (both
# into the build tree and into lib/cmake/cudaq on install), so this is a lookup
# of what the library actually contains rather than a re-derivation of it.
function(_cudaq_read_mlir_provided_libs _out_var)
  set(_provided)
  get_property(_bundle GLOBAL PROPERTY CUDAQ_MLIR_BUNDLE_LIBS)
  if(_bundle)
    list(APPEND _provided ${_bundle})
  endif()

  # In-tree the manifest sits in the build tree; installed it sits next to this
  # file in lib/cmake/cudaq.
  set(_manifest_candidates
    "${CMAKE_BINARY_DIR}/lib/cmake/cudaq/cudaqMLIR-contents.txt"
    "${CMAKE_CURRENT_LIST_DIR}/cudaqMLIR-contents.txt")
  foreach(_candidate IN LISTS _manifest_candidates)
    if(EXISTS "${_candidate}")
      cudaq_read_symbol_list("${_candidate}" _contents)
      list(APPEND _provided ${_contents})
      break()
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _provided)
  set(${_out_var} "${_provided}" PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------- #
# ``cudaq_check_mlir_symbol_closure(<target>)``
#
# Fail the build if ``<target>`` references MLIR/LLVM symbols that libcudaqMLIR
# does not export. Everything in CUDA-Q resolves MLIR/LLVM dynamically from the
# single libcudaqMLIR instance, and a gap there is not a link error -- it
# surfaces as a dlopen/ImportError much later, typically in a wheel-validation
# job. See scripts/check_mlir_symbols.sh.
# --------------------------------------------------------------------------- #

# nm invocations are only exercised on Linux; leave the check opt-in elsewhere.
if(APPLE)
  set(_cudaq_symbol_closure_default OFF)
else()
  set(_cudaq_symbol_closure_default ON)
endif()
option(CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE
  "Fail the build when a library references MLIR/LLVM symbols libcudaqMLIR does not export."
  ${_cudaq_symbol_closure_default})

# In-tree this module sits in cmake/modules; installed it sits in
# lib/cmake/cudaq next to a copy of the script.
foreach(_candidate
    "${CMAKE_CURRENT_LIST_DIR}/../../scripts/check_mlir_symbols.sh"
    "${CMAKE_CURRENT_LIST_DIR}/check_mlir_symbols.sh")
  if(EXISTS "${_candidate}")
    get_filename_component(CUDAQ_SYMBOL_CHECK_SCRIPT "${_candidate}" ABSOLUTE)
    break()
  endif()
endforeach()

function(cudaq_check_mlir_symbol_closure name)
  if(NOT CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE OR NOT CUDAQ_SYMBOL_CHECK_SCRIPT)
    return()
  endif()
  add_custom_command(TARGET ${name} POST_BUILD
    COMMAND bash "${CUDAQ_SYMBOL_CHECK_SCRIPT}"
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

  # Check up front: without ENABLE_AGGREGATION there is no obj.<lib> to embed,
  # and the failure otherwise surfaces as an empty generator expression.
  foreach(_capi_lib IN LISTS _embed_libs)
    if(NOT TARGET obj.${_capi_lib})
      message(FATAL_ERROR "Ensure ${_capi_lib} was registered with ENABLE_AGGREGATION")
    endif()
  endforeach()

  # 3. Create the shared library, with hidden visibility and linking to cudaqMLIR
  #
  # add_mlir_aggregate() embeds obj.<lib> for every EMBED_LIBS entry and links
  # the dependencies those libraries advertise. Upstream rewrites those
  # dependency lists into generator expressions that drop any entry named in
  # the consuming target's MLIR_AGGREGATE_EXCLUDE_LIBS property -- see
  # get_mlir_filtered_link_libraries() in AddMLIR.cmake. add_mlir_aggregate
  # seeds that property with the embedded libraries; appending the contents of
  # libcudaqMLIR afterwards drops every static MLIR/LLVM archive as well,
  # because those generator expressions are not evaluated until generate time.
  # What survives is exactly the project-owned dependencies -- e.g. a
  # downstream project's dialect libraries -- which do have to be linked.
  #
  # This also gets us upstream's "LINKER:-z,defs", so an unresolved symbol here
  # is a link error rather than a dlopen failure at run time.
  add_mlir_aggregate(${name}
    SHARED
    DISABLE_INSTALL
    EMBED_LIBS ${_embed_libs}
    PUBLIC_LIBS cudaq::cudaqMLIR)

  _cudaq_read_mlir_provided_libs(_mlir_provided)
  set_property(TARGET ${name} APPEND PROPERTY
    MLIR_AGGREGATE_EXCLUDE_LIBS ${_mlir_provided})

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

  cudaq_check_mlir_symbol_closure(${name})

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

  # The extension modules install to cudaq/mlir/_mlir_libs, whereas the CUDA-Q
  # runtime libraries (libcudaq-common, libnvqir, libcudaqMLIR, cudaqMLIRCAPI,
  # and the dlopen'd simulator plugins) live three levels up under lib/.
  # INSTALL_RPATH must therefore be a *relative* ($ORIGIN / @loader_path) path
  # so the installed and wheel-repaired modules can resolve those libraries at
  # load time. Using the absolute build directory (CUDAQ_LIBRARY_DIR) here is
  # wrong: it is stripped by auditwheel/delocate, which then treats the CUDA-Q
  # libraries as external and grafts (and mangles) them into a separate .libs
  # directory. That moves libcudaq-common away from the simulator plugins, so
  # getCUDAQLibraryPath() resolves to .libs/ and the plugins (which stay in
  # lib/) can no longer be found ("target qpp-cpu doesn't define a simulator" /
  # "Could not load the requested plugin"). Keeping a relative rpath into lib/
  # lets auditwheel find them in-wheel and leave them in place.
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
  target_link_options(${name} BEFORE PRIVATE "$<TARGET_FILE:cudaq::cudaqMLIR>")
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

# ``cudaq::MLIR`` is the public name downstream projects link against; it is an
# alias for the ``cudaq::cudaqMLIR`` target, which is what the export set and
# the in-tree build define. GLOBAL so it is usable from any subdirectory, and
# guarded so that including this module twice is harmless.
if(NOT TARGET cudaq::MLIR)
  add_library(cudaq::MLIR INTERFACE IMPORTED GLOBAL)
  set_target_properties(cudaq::MLIR PROPERTIES
    INTERFACE_LINK_LIBRARIES cudaq::cudaqMLIR
  )
endif()
