# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Private third-party dependencies of the CUDAQ target component.
#
# yaml-cpp 0.9.0 and reflect-cpp v0.25.0 are populated with FetchContent from
# immutable upstream commits and are built as private static libraries.

include(FetchContent)

FetchContent_Declare(
  yaml_cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  # yaml-cpp 0.9.0
  GIT_TAG 56e3bb550c91fd7005566f19c079cb7a503223cf
  GIT_SUBMODULES ""
  EXCLUDE_FROM_ALL
)
FetchContent_Declare(
  reflectcpp
  GIT_REPOSITORY https://github.com/getml/reflect-cpp.git
  # reflect-cpp
  GIT_TAG 1327ef1d0b403f1424776da4c47bf0c7a6b0feaa
  GIT_SUBMODULES ""
  EXCLUDE_FROM_ALL
)

# yaml-cpp: core static library, none of the optional parts.
set(YAML_BUILD_SHARED_LIBS OFF)
set(YAML_ENABLE_PIC ON)
set(YAML_CPP_INSTALL OFF)
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
set(YAML_CPP_BUILD_CONTRIB OFF)
set(YAML_CPP_FORMAT_SOURCE OFF)

# reflect-cpp: YAML support only, on top of the bundled header dependencies
# (ctre, enchantum); never resolve dependencies through vcpkg.
set(REFLECTCPP_BUILD_SHARED OFF)
set(REFLECTCPP_INSTALL OFF)
set(REFLECTCPP_YAML ON)
set(REFLECTCPP_JSON OFF)
set(REFLECTCPP_AVRO OFF)
set(REFLECTCPP_BSON OFF)
set(REFLECTCPP_CAPNPROTO OFF)
set(REFLECTCPP_CBOR OFF)
set(REFLECTCPP_CEREAL OFF)
set(REFLECTCPP_CSV OFF)
set(REFLECTCPP_FLEXBUFFERS OFF)
set(REFLECTCPP_MSGPACK OFF)
set(REFLECTCPP_PARQUET OFF)
set(REFLECTCPP_XML OFF)
set(REFLECTCPP_TOML OFF)
set(REFLECTCPP_UBJSON OFF)
set(REFLECTCPP_YAS OFF)
set(REFLECTCPP_BOOST_SERIALIZATION OFF)
set(REFLECTCPP_USE_VCPKG OFF)
set(REFLECTCPP_USE_BUNDLED_DEPENDENCIES ON)
set(REFLECTCPP_BUILD_TESTS OFF)
set(REFLECTCPP_BUILD_BENCHMARKS OFF)
set(REFLECTCPP_CHECK_HEADERS OFF)

# Ordering matters: yaml-cpp is a dependency of reflect-cpp.
FetchContent_MakeAvailable(yaml_cpp reflectcpp)

# Private static libraries, with their symbols hidden from the shared library
# that links them. SYSTEM keeps the upstream headers out of reach of the
# warnings-as-errors settings the root build applies to our own sources.
foreach(_cudaq_target_dep IN ITEMS reflectcpp yaml-cpp)
  if (TARGET ${_cudaq_target_dep})
    set_target_properties(${_cudaq_target_dep} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      COMPILE_WARNING_AS_ERROR OFF
      SYSTEM ON
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON)
  endif()
endforeach()
