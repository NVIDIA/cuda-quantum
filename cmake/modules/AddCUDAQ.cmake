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

# Replicate all build configs on `${name}` to `obj.${name}`
function(_cudaq_forward_object_usage_requirements name)
  if(NOT TARGET obj.${name})
    return()
  endif()
  target_include_directories(obj.${name} SYSTEM PRIVATE
    $<TARGET_PROPERTY:${name},INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(obj.${name} PRIVATE
    $<TARGET_PROPERTY:${name},INTERFACE_COMPILE_DEFINITIONS>)
  target_compile_options(obj.${name} PRIVATE
    $<TARGET_PROPERTY:${name},INTERFACE_COMPILE_OPTIONS>)
endfunction()

# target_link_libraries() for a library created by add_cudaq_library().  This ensures
# the dependency is also set on obj.<target>, (used to build the `cudaqMLIR` shared library.
function(cudaq_target_link_libraries target visibility)
  target_link_libraries(${target} ${visibility} ${ARGN})
  if(TARGET obj.${target})
    target_link_libraries(obj.${target} PRIVATE ${ARGN})
  endif()
endfunction()

function(add_cudaq_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL ENABLE_AGGREGATION)
  add_cudaq_library_install(${name})
  _cudaq_forward_object_usage_requirements(${name})
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
# ``cudaq_check_mlir_symbol_closure(<target> [PROVIDERS <target>...])``
#
# Fail the build if
#  - ``<target>`` references MLIR/LLVM symbols that neither `libcudaqMLIR` nor
#    any of the additional `PROVIDERS` export, or
#  - re-defines duplicate (strong) symbols already defined in `libcudaqMLIR`.
#
# This wraps `scripts/check_mlir_symbols.sh`. See the script for more details.
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

if(CMAKE_NM AND NOT CUDAQ_NM)
  set(CUDAQ_NM "${CMAKE_NM}" CACHE FILEPATH
    "nm used to verify the MLIR/LLVM symbol closure")
endif()
find_program(CUDAQ_NM
  NAMES nm llvm-nm
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "$ENV{LLVM_INSTALL_PREFIX}/bin"
  DOC "nm used to verify the MLIR/LLVM symbol closure")

if(CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE AND NOT CUDAQ_NM)
  message(STATUS
    "Neither nm nor llvm-nm found: skipping the MLIR/LLVM symbol closure check.")
endif()

function(cudaq_check_mlir_symbol_closure name)
  cmake_parse_arguments(ARG "" "" "PROVIDERS" ${ARGN})
  if(NOT CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE OR NOT CUDAQ_CHECK_SYMBOL_SCRIPT
      OR NOT CUDAQ_NM)
    return()
  endif()
  set(_providers)
  foreach(_provider IN LISTS ARG_PROVIDERS)
    if(TARGET ${_provider})
      list(APPEND _providers "$<TARGET_FILE:${_provider}>")
    endif()
  endforeach()
  add_custom_command(TARGET ${name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env "NM=${CUDAQ_NM}"
    bash "${CUDAQ_CHECK_SYMBOL_SCRIPT}"
    "$<TARGET_FILE:${name}>" "$<TARGET_FILE:cudaq::cudaqMLIR>" ${_providers}
    COMMENT "Checking MLIR/LLVM symbol closure of ${name}"
    VERBATIM)
endfunction()

# CUDAQ_PYTHON_BINDINGS_SHARED_LIBS controls whether the common CAPI
# aggregate built by add_cudaq_python_common_capi_library() (below) is a
# shared or static library. It defaults to ON: the aggregate is loaded
# directly by a Python interpreter via the nanobind extension modules, so a
# shared library is the correct default. A build engineer embedding these
# bindings into a fully static, custom Python interpreter (or otherwise
# assembling their own deployment) can flip this OFF.
option(CUDAQ_PYTHON_BINDINGS_SHARED_LIBS
  "Build the cudaq/ Python bindings' common CAPI library as a shared library."
  ON)

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

  # 3. Create the library, with hidden visibility and linking to cudaqMLIR
  #
  # We use the MLIR-provided aggregation utility but modify it to exclude any
  # libraries provided by `libcudaqMLIR.so` and instead link in `cudaq::cudaqMLIR`.
  # We then hide all symbols by default (same as add_mlir_python_common_capi_library).
  #
  # SHARED by default (CUDAQ_PYTHON_BINDINGS_SHARED_LIBS): this is what a
  # Python interpreter dlopen()s. See that option's docstring for the STATIC
  # override use case.
  if(CUDAQ_PYTHON_BINDINGS_SHARED_LIBS)
    set(_cudaq_python_capi_libtype SHARED)
  else()
    set(_cudaq_python_capi_libtype STATIC)
  endif()
  add_mlir_aggregate(${name}
    ${_cudaq_python_capi_libtype}
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
#   - links ``cudaq::cudaqMLIR`` so MLIR/LLVM symbols resolve from the wheel
#     dylib rather than from static component archives (link order comes from
#     ``cudaq::cudaqMLIR``'s ``INTERFACE_LINK_OPTIONS``).
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

  cmake_parse_arguments(ARG "" "" "COMMON_CAPI_LINK_LIBS" ${ARGN})
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

    cudaq_check_mlir_symbol_closure(${_dso} PROVIDERS ${ARG_COMMON_CAPI_LINK_LIBS})
  endforeach()
endfunction()

# Adds a CUDA Quantum dialect library target for installation. This should normally
# only be called from add_cudaq_library().
#
# <name> will be registered as part of the `cudaq-dev-targets` export set.
function(add_cudaq_library_install name)
  install(TARGETS ${name} COMPONENT Development EXPORT cudaq-dev-targets)
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

# CUDAQ_INSTALL_TARGET_YAML controls whether each target's raw `.yml` file is
# installed. While it is no longer required under the normal flow of the tools,
# one can still install the .yml files to support legacy dependences.
option(CUDAQ_INSTALL_TARGET_YAML
  "Install each target's raw .yml file to <install>/targets/. Not required \
for in-tree targets to resolve correctly at runtime, in Python, or via \
nvq++ - only external/plugin targets require an on-disk .yml." ON)

function(_cudaq_resolve_target_yml name out_var)
  set(_local ${CMAKE_CURRENT_SOURCE_DIR}/${name}.yml)
  if (EXISTS ${_local})
    set(${out_var} ${_local} PARENT_SCOPE)
  else()
    set(${out_var} ${CMAKE_SOURCE_DIR}/cudaq/lib/Targets/${name}/${name}.yml
        PARENT_SCOPE)
  endif()
endfunction()

function(add_target_config name)
  _cudaq_resolve_target_yml(${name} _yml)
  if (CUDAQ_INSTALL_TARGET_YAML)
    install(FILES ${_yml} DESTINATION targets COMPONENT Runtime)
    configure_file(${_yml} ${CMAKE_BINARY_DIR}/targets/${name}.yml COPYONLY)
  endif()
  set_property(GLOBAL APPEND PROPERTY CUDAQ_TARGET_DB_NAMES ${name})
  set_property(GLOBAL APPEND PROPERTY CUDAQ_TARGET_DB_PATHS ${_yml})
endfunction()

# Generates the precompiled target database (TargetDatabase.gen.cpp) from every
# `.yml` registered via add_target_config() so far, and builds it into the
# CUDAQTargetDatabase library. Must be called once in the build, after every
# add_target_config() call site has been processed.
function(cudaq_finalize_target_database)
  get_property(_names GLOBAL PROPERTY CUDAQ_TARGET_DB_NAMES)
  get_property(_paths GLOBAL PROPERTY CUDAQ_TARGET_DB_PATHS)
  list(LENGTH _names _count)

  set(_gen_args)
  set(_deps)
  math(EXPR _lastIdx "${_count} - 1")
  foreach(_idx RANGE ${_lastIdx})
    list(GET _names ${_idx} _name)
    list(GET _paths ${_idx} _path)
    list(APPEND _gen_args "${_name}=${_path}")
    list(APPEND _deps ${_path})
  endforeach()

  set(_gen_cpp ${CMAKE_BINARY_DIR}/generated/TargetDatabase.gen.cpp)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)
  add_custom_command(
    OUTPUT ${_gen_cpp}
    COMMAND $<TARGET_FILE:cudaq-target-db-gen> -o ${_gen_cpp} ${_gen_args}
    DEPENDS cudaq-target-db-gen ${_deps}
    COMMENT "Generating precompiled CUDA-Q target database (${_count} targets)"
    VERBATIM)

  add_library(CUDAQTargetDatabase STATIC
    ${_gen_cpp}
    ${CMAKE_SOURCE_DIR}/cudaq/lib/Target/TargetDatabase.cpp)
  target_link_libraries(CUDAQTargetDatabase PUBLIC CUDAQTargetConfig)
  install(TARGETS CUDAQTargetDatabase
          EXPORT cudaq-common-targets
          DESTINATION lib
          COMPONENT Runtime)

  # nvq++ (configured output) contains a literal
  # `__CUDAQ_BUILTIN_TARGET_NAMES__` placeholder. Patch it once more, with
  # exactly this build's enabled target set.
  set(_nvqpp_script ${CUDAQ_BINARY_DIR}/bin/nvq++)
  if (EXISTS ${_nvqpp_script})
    list(TRANSFORM _names REPLACE "(.+)" "\"\\1\"" OUTPUT_VARIABLE _quoted_names)
    list(JOIN _quoted_names " " _names_bash_array)
    file(READ ${_nvqpp_script} _nvqpp_contents)
    string(REPLACE "__CUDAQ_BUILTIN_TARGET_NAMES__" "${_names_bash_array}"
      _nvqpp_contents "${_nvqpp_contents}")
    file(WRITE ${_nvqpp_script} "${_nvqpp_contents}")
  endif()
endfunction()

# Generates cudaq/'s OWN precompiled target database - a separate artifact
# from cudaq_finalize_target_database() above (which is runtime/'s, built
# from its own conditional add_target_config() calls). cudaq/ must never
# depend on runtime/, but cudaq-opt and cudaq-target-resolve do need target
# data.
#
# The database is sourced from the union of:
#  (a) every .yml file physically present under cudaq/lib/Targets/; and
#  (b) any add_target_config() registration whose resolved .yml is NOT
#      under cudaq/lib/Targets/ including a genuine external registration (a
#      CUDAQ_EXTERNAL_PROJECTS entry calling add_target_config() from its
#      own, out-of-tree CMakeLists.txt.
#
# Must therefore be called after every add_target_config() call site - the same
# timing constraint as cudaq_finalize_target_database() - even though the
# library it defines (CUDAQBuiltinTargetDb) is consumed by cudaq-opt and
# cudaq-target-resolve, which are built earlier: CMake resolves
# target_link_libraries() target names at generate time, not in
# add_subdirectory() order, so this is safe.
function(cudaq_finalize_builtin_target_database)
  file(GLOB _target_dirs LIST_DIRECTORIES true
    ${CMAKE_SOURCE_DIR}/cudaq/lib/Targets/*)
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
      string(FIND "${_path}" "${CMAKE_SOURCE_DIR}/cudaq/lib/Targets/" _idx_pos)
      if (_idx_pos EQUAL 0)
        continue() # already covered by the static glob above
      endif()
      list(APPEND _gen_args "${_name}=${_path}")
      list(APPEND _deps ${_path})
      list(APPEND _seen_names ${_name})
      math(EXPR _count "${_count} + 1")
    endforeach()
  endif()

  set(_gen_cpp ${CMAKE_BINARY_DIR}/generated/CudaqBuiltinTargetDatabase.gen.cpp)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)
  add_custom_command(
    OUTPUT ${_gen_cpp}
    COMMAND $<TARGET_FILE:cudaq-target-db-gen> -o ${_gen_cpp} ${_gen_args}
    DEPENDS cudaq-target-db-gen ${_deps}
    COMMENT "Generating precompiled cudaq/ target database (${_count} targets)"
    VERBATIM)

  add_library(CUDAQBuiltinTargetDb STATIC
    ${_gen_cpp}
    ${CMAKE_SOURCE_DIR}/cudaq/lib/Target/TargetDatabase.cpp)
  target_link_libraries(CUDAQBuiltinTargetDb PUBLIC CUDAQTargetConfig)
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

# Define the CUDAQ dev targets for downstream projects when they exist.
if(NOT TARGET QuakeDialect
    AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/CUDAQDevTargets.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/CUDAQDevTargets.cmake")
endif()

# Define the public alias ``cudaq::MLIR`` for use in downstream projects.
if(NOT TARGET cudaq::MLIR)
  add_library(cudaq::MLIR INTERFACE IMPORTED GLOBAL)
  set_target_properties(cudaq::MLIR PROPERTIES
    INTERFACE_LINK_LIBRARIES cudaq::cudaqMLIR
  )
  # Also expose the header files through `cudaq::MLIR`
  if(CUDAQ_INCLUDE_DIR AND IS_DIRECTORY "${CUDAQ_INCLUDE_DIR}")
    set_target_properties(cudaq::MLIR PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAQ_INCLUDE_DIR}"
    )
  endif()
endif()
