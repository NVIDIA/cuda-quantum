# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Configure the real project in isolated build trees; no compilation is needed.
cmake_minimum_required(VERSION 3.30)

set(failures "")
function(check_config name expect_failure projects)
  set(build_dir "${BINARY_DIR}/${name}")
  file(MAKE_DIRECTORY "${build_dir}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --fresh -S "${SOURCE_DIR}" -B "${build_dir}"
      "-DLLVM_DIR=${LLVM_DIR}" "-DMLIR_DIR=${MLIR_DIR}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
      "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}"
      "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
      "-DBLAS_ROOT=${BLAS_ROOT}" "-DBLAS_LIBRARIES=${BLAS_LIBRARIES}"
      "-DBLA_VENDOR=${BLA_VENDOR}" "-DBLA_STATIC=${BLA_STATIC}"
      "-DBLA_SIZEOF_INTEGER=${BLA_SIZEOF_INTEGER}"
      "-DBLA_PREFER_PKGCONFIG=${BLA_PREFER_PKGCONFIG}"
      "-DBLA_PKGCONFIG_BLAS=${BLA_PKGCONFIG_BLAS}"
      "-Dnanobind_DIR=${nanobind_DIR}"
      "-DZLIB_ROOT=${ZLIB_ROOT}" "-DGMP_ROOT=${GMP_ROOT}"
      "-DMPFR_ROOT=${MPFR_ROOT}" "-DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}"
      "-DCURL_ROOT=${CURL_ROOT}"
      -DGIT_SUBMODULE=OFF
      -DCUDAQ_BUILD_TESTS=OFF
      -DCUDAQ_DISABLE_CPP_FRONTEND=ON -DCUDAQ_DISABLE_TOOLS=ON
      -DCUDAQ_ENABLE_ALL_BACKEND=OFF -DCMAKE_CUDA_COMPILER=NOTFOUND
      -DCUDAQ_SKIP_MPI=ON
      "-DCUDAQ_ENABLE_PROJECTS=${projects}" -DCUDAQ_ENABLE_PYTHON_BINDINGS=ON
      -DCUDAQ_ENABLE_ENSMALLEN=ON -DCUDAQ_ENABLE_NLOPT=ON
      -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=OFF
      ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  file(WRITE "${build_dir}/configure.log" "${output}${error}")
  string(REGEX REPLACE "[ \t\r\n]+" " " diagnostic "${output}${error}")
  if(expect_failure)
    if(result EQUAL 0 OR NOT diagnostic MATCHES
        "Python runtime requires CUDAQ_ENABLE_ENSMALLEN=ON and CUDAQ_ENABLE_NLOPT=ON")
      list(APPEND failures "${name}")
      set(failures "${failures}" PARENT_SCOPE)
      message(STATUS "FAIL: ${name}: expected optimizer diagnostic; see ${build_dir}/configure.log")
      return()
    endif()
  elseif(NOT result EQUAL 0)
    list(APPEND failures "${name}")
    set(failures "${failures}" PARENT_SCOPE)
    message(STATUS "FAIL: ${name}: configure returned ${result}; see ${build_dir}/configure.log")
    return()
  endif()
  message(STATUS "PASS: ${name}")
endfunction()

check_config(python_enabled FALSE python)
check_config(python_no_ensmallen TRUE python -DCUDAQ_ENABLE_ENSMALLEN=OFF)
check_config(python_no_nlopt TRUE python -DCUDAQ_ENABLE_NLOPT=OFF)
check_config(python_no_optimizers TRUE python
  -DCUDAQ_ENABLE_ENSMALLEN=OFF -DCUDAQ_ENABLE_NLOPT=OFF)
check_config(python_no_blas TRUE python -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON)
check_config(legacy_python_no_ensmallen TRUE ""
  -DCUDAQ_ENABLE_PYTHON=ON
  -DCUDAQ_ENABLE_ENSMALLEN=OFF)
check_config(runtime_no_optimizers FALSE "cudaq;runtime"
  -DCUDAQ_ENABLE_PYTHON_BINDINGS=OFF
  -DCUDAQ_ENABLE_ENSMALLEN=OFF -DCUDAQ_ENABLE_NLOPT=OFF)
check_config(runtime_no_blas FALSE "cudaq;runtime"
  -DCUDAQ_ENABLE_PYTHON_BINDINGS=OFF
  -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON)
check_config(core_bindings_no_optimizers FALSE cudaq
  -DCUDAQ_ENABLE_ENSMALLEN=OFF -DCUDAQ_ENABLE_NLOPT=OFF)
check_config(runtime_with_legacy_python_flag FALSE "cudaq;runtime"
  -DCUDAQ_ENABLE_PYTHON=ON
  -DCUDAQ_ENABLE_PYTHON_BINDINGS=OFF
  -DCUDAQ_ENABLE_ENSMALLEN=OFF -DCUDAQ_ENABLE_NLOPT=OFF)

if(failures)
  message(FATAL_ERROR "Optimizer configuration regressions: ${failures}")
endif()
