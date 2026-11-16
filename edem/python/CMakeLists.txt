# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
set(CUDAQ_EDEM_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" PARENT_SCOPE)
if(NOT nanobind_DIR)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
# CUDA-Q's vendored nanobind has no CMake package-version file.
find_package(nanobind CONFIG REQUIRED)

if(SKBUILD)
  # The wheel stages the entire native installation under cudaq_edem/.
  set(python_install_dir .)
else()
  set(CUDAQ_EDEM_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/cudaq_edem"
    CACHE STRING "Install directory for cudaq_edem, relative to CMAKE_INSTALL_PREFIX")
  set(python_install_dir "${CUDAQ_EDEM_PYTHON_INSTALL_DIR}")
endif()
set(CUDAQ_EDEM_PYTHON_INSTALL_DIR "${python_install_dir}" PARENT_SCOPE)
file(RELATIVE_PATH CUDAQ_EDEM_PYTHON_CMAKE_DIR
  "${CMAKE_INSTALL_PREFIX}/${python_install_dir}"
  "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/cudaq-edem")
file(RELATIVE_PATH python_library_dir
  "${CMAKE_INSTALL_PREFIX}/${python_install_dir}"
  "${CMAKE_INSTALL_FULL_LIBDIR}")

nanobind_add_module(cudaq-edem-python NB_STATIC NB_DOMAIN cudaq_edem_v0 bindings.cpp)
target_link_libraries(cudaq-edem-python PRIVATE cudaq::edem)
set_target_properties(cudaq-edem-python PROPERTIES
  OUTPUT_NAME _core
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/cudaq_edem"
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH_USE_LINK_PATH FALSE)
if(CUDAQ_EDEM_BUILD_TESTS)
  nanobind_add_module(cudaq-edem-adoption-test NB_STATIC
    NB_DOMAIN cudaq_edem_adoption_test_v0 adoption_test.cpp)
  target_link_libraries(cudaq-edem-adoption-test PRIVATE cudaq::edem)
  set_target_properties(cudaq-edem-adoption-test PROPERTIES
    OUTPUT_NAME _adoption_test
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/cudaq_edem")
endif()
if(APPLE)
  set_property(TARGET cudaq-edem-python PROPERTY
    INSTALL_RPATH "@loader_path/${python_library_dir}")
elseif(UNIX)
  set_property(TARGET cudaq-edem-python PROPERTY
    INSTALL_RPATH "$ORIGIN/${python_library_dir}")
endif()

configure_file(cudaq_edem/__init__.py __init__.py @ONLY)
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/cudaq_edem/__init__.py"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
install(TARGETS cudaq-edem-python
  LIBRARY DESTINATION "${python_install_dir}" COMPONENT EDEMPython)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
  DESTINATION "${python_install_dir}" COMPONENT EDEMPython)
install(FILES ../../LICENSES/LICENSE.nanobind ../../LICENSES/LICENSE.robin-map
  DESTINATION "${CMAKE_INSTALL_DATADIR}/cudaq-edem/licenses" COMPONENT EDEMPython)
