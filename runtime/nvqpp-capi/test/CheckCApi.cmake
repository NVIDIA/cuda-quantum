# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Checks for the nvq++ C API library:
#  1. The shared library exports exactly the symbols declared in nvqpp_capi.h.
#  2. The static archive defines all of them.
#  3. The static archive links against the runtime (which also checks that the
#     runtime implements the ABI version that the archive expects).
#
# Expects: NM, CXX, CAPI, CAPI_STATIC, RUNTIME, EXPORTS, WORK_DIR,
# APPLE.

set(expected ${EXPORTS})
list(SORT expected)

# Run nm and return the symbol names (without the Mach-O leading underscore).
function(nm_symbols out)
  execute_process(COMMAND ${NM} ${ARGN} OUTPUT_VARIABLE raw
                  RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${NM} ${ARGN} failed: ${err}")
  endif()
  string(REPLACE "\n" ";" lines "${raw}")
  set(syms "")
  foreach(line IN LISTS lines)
    # Skip archive member headers and blank lines.
    if(line STREQUAL "" OR line MATCHES ":$")
      continue()
    endif()
    string(REGEX REPLACE "^.* " "" sym "${line}")
    # Mach-O symbols carry one extra leading underscore. (Not REGEX REPLACE:
    # it would re-apply `^_` after each replacement.)
    if(APPLE AND sym MATCHES "^_")
      string(SUBSTRING "${sym}" 1 -1 sym)
    endif()
    list(APPEND syms "${sym}")
  endforeach()
  list(REMOVE_DUPLICATES syms)
  list(SORT syms)
  set(${out} "${syms}" PARENT_SCOPE)
endfunction()

# 1. Exact exports of the shared library.
if(APPLE)
  nm_symbols(exported -g --defined-only ${CAPI})
else()
  nm_symbols(exported -D -g --defined-only ${CAPI})
  # Linker-provided ELF symbols, if any.
  list(REMOVE_ITEM exported _init _fini _edata _end __bss_start)
endif()
if(NOT exported STREQUAL expected)
  set(extra ${exported})
  list(REMOVE_ITEM extra ${expected})
  set(missing ${expected})
  list(REMOVE_ITEM missing ${exported})
  message(FATAL_ERROR "C API exports do not match nvqpp_capi.h.\n"
                      "Unexpected: ${extra}\nMissing: ${missing}")
endif()

# 2. The static archive defines the whole API.
nm_symbols(archive_defined -g --defined-only ${CAPI_STATIC})
set(missing ${expected})
list(REMOVE_ITEM missing ${archive_defined})
if(missing)
  message(FATAL_ERROR "Static C API library does not define: ${missing}")
endif()

# 3. The archive links against the runtime. Unresolved symbols, including the
# runtime ABI version symbol, make this fail.
file(MAKE_DIRECTORY ${WORK_DIR})
file(WRITE ${WORK_DIR}/main.cpp "int main() {}\n")
get_filename_component(runtime_dir ${RUNTIME} DIRECTORY)
set(link_flags -L${runtime_dir} -lcudaq -lcudaq-common
               -Wl,-rpath,${runtime_dir})
if(APPLE)
  set(archive -Wl,-force_load,${CAPI_STATIC})
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE sdk
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(sdk)
    list(APPEND link_flags -isysroot ${sdk})
  endif()
else()
  set(archive -Wl,--whole-archive ${CAPI_STATIC} -Wl,--no-whole-archive)
  list(APPEND link_flags -Wl,-rpath-link,${runtime_dir})
endif()
execute_process(
  COMMAND ${CXX} ${WORK_DIR}/main.cpp ${archive} -o ${WORK_DIR}/main
          ${link_flags}
  RESULT_VARIABLE rc OUTPUT_QUIET ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "Linking the static C API library failed:\n${err}")
endif()

message(STATUS "nvq++ C API checks passed.")
