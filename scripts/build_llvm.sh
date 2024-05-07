#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts builds the clang and mlir project from the source in the LLVM submodule.
# The binaries will be installed in the folder defined by the LLVM_INSTALL_PREFIX environment
# variable, or in $HOME/.llvm if LLVM_INSTALL_PREFIX is not defined.
# If Python bindings are generated, pybind11 will be built and installed in the location 
# defined by PYBIND11_INSTALL_PREFIX unless that folder already exists.
#
# Usage:
# bash scripts/build_llvm.sh
# -or-
# bash scripts/build_llvm.sh -c Debug
# -or-
# LLVM_INSTALL_PREFIX=/installation/path/ bash scripts/build_llvm.sh
#
# For documentation on how to assemble a complete toolchain, multi-stage builds,
# and OpenMP support within Clang, see
# - https://clang.llvm.org/docs/Toolchain.html
# - https://llvm.org/docs/AdvancedBuilds.html
# - https://github.com/llvm/llvm-project/tree/main/clang/cmake/caches
# - https://github.com/llvm/llvm-project/blob/main/openmp/docs/SupportAndFAQ.rst#q-how-to-build-an-openmp-gpu-offload-capable-compiler

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-$HOME/.llvm}
LLVM_PROJECTS=${LLVM_PROJECTS:-'clang;lld;mlir;python-bindings'}
PYBIND11_INSTALL_PREFIX=${PYBIND11_INSTALL_PREFIX:-/usr/local/pybind11}
Python3_EXECUTABLE=${Python3_EXECUTABLE:-python3}

# Process command line arguments.
build_configuration=Release
verbose=false

__optind__=$OPTIND
OPTIND=1
while getopts ":c:v" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
    ;;
    v) verbose=true
    ;;
    :) echo "Option -$OPTARG requires an argument."
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
echo "Configured C compiler: $CC"
echo "Configured C++ compiler: $CXX"

# Check if we build python bindings and build pybind11 from source if necessary.
projects=(`echo $LLVM_PROJECTS | tr ';' ' '`)
llvm_projects=`printf "%s;" "${projects[@]}"`
if [ -z "${llvm_projects##*python-bindings;*}" ]; then
  mlir_python_bindings=ON
  projects=("${projects[@]/python-bindings}")

  if [ ! -d "$PYBIND11_INSTALL_PREFIX" ] || [ -z "$(ls -A "$PYBIND11_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
    cd "$this_file_dir" && cd $(git rev-parse --show-toplevel)
    echo "Building PyBind11..."
    git submodule update --init --recursive --recommend-shallow --single-branch tpls/pybind11 
    mkdir "tpls/pybind11/build" && cd "tpls/pybind11/build"
    cmake -G Ninja ../ -DCMAKE_INSTALL_PREFIX="$PYBIND11_INSTALL_PREFIX" -DPYBIND11_TEST=False
    cmake --build . --target install --config Release
  fi
fi

# Prepare the source and build directory.
if [ ! -d "$LLVM_SOURCE" ] || [ -z "$(ls -A "$LLVM_SOURCE"/* 2> /dev/null)" ]; then
  echo "Cloning LLVM submodule..."
  cd "$this_file_dir" && cd $(git rev-parse --show-toplevel)
  LLVM_SOURCE="${LLVM_SOURCE:-$HOME/.llvm-project}"
  llvm_repo="$(git config --file=.gitmodules submodule.tpls/llvm.url)"
  llvm_commit="$(git submodule | grep tpls/llvm | cut -c2- | cut -d ' ' -f1)"
  git clone --filter=tree:0 "$llvm_repo" "$LLVM_SOURCE"
  cd "$LLVM_SOURCE" && git checkout $llvm_commit

  LLVM_CMAKE_PATCHES=${LLVM_CMAKE_PATCHES:-"$this_file_dir/../tpls/customizations/llvm"}
  if [ -d "$LLVM_CMAKE_PATCHES" ]; then 
    echo "Applying LLVM patches in $LLVM_CMAKE_PATCHES..."
    for patch in `find "$LLVM_CMAKE_PATCHES"/* -maxdepth 0 -type f -name '*.diff'`; do
      # Check if patch is already applied.
      git apply "$patch" --ignore-whitespace --reverse --check
      if [ ! 0 -eq $? ]; then
        # If the patch is not yet applied, apply the patch.
        git apply "$patch" --ignore-whitespace
        if [ ! 0 -eq $? ]; then
          echo "Applying patch $patch failed. Please update patch."
          (return 0 2>/dev/null) && return 1 || exit 1
        fi
      fi
    done
  fi
fi

mkdir -p "$LLVM_INSTALL_PREFIX"
mkdir -p "$LLVM_SOURCE/${LLVM_BUILD_FOLDER:-build}"
cd "$LLVM_SOURCE/${LLVM_BUILD_FOLDER:-build}"
mkdir -p logs && rm -rf logs/* 

# Specify which components we need to keep the size of the LLVM build down.
# To get a list of install targets, check the output of the following command 
# in the build folder:
#   ninja -t targets | grep -Po 'install-\K.*(?=-stripped:)'
echo "Preparing LLVM build..."
if [ -z "${llvm_projects##*runtimes;*}" ]; then
  echo "- including runtime components"
  llvm_runtimes+="libcxx;libcxxabi;libunwind;compiler-rt;"
  projects=("${projects[@]/runtimes}")
fi
if [ -z "${llvm_projects##*openmp;*}" ]; then
  # Enabled separately from other runtimes, since we need to build the other
  # runtime libraries first before being able to build openmp.
  echo "- including OpenMP runtime"
  llvm_runtimes+="openmp;"
  projects=("${projects[@]/openmp}")
fi

llvm_projects=`printf "%s;" "${projects[@]}"`
if [ -z "${llvm_projects##*clang;*}" ]; then
  echo "- including Clang components"
  llvm_components+="clang;clang-format;clang-cmake-exports;clang-headers;clang-libraries;clang-resource-headers;"
  projects=("${projects[@]/clang}")
fi
if [ -z "${llvm_projects##*flang;*}" ]; then
  echo "- including Flang components"
  llvm_components+="flang-new;"
  projects=("${projects[@]/flang}")
fi
if [ -z "${llvm_projects##*mlir;*}" ]; then
  echo "- including MLIR components"
  llvm_components+="mlir-cmake-exports;mlir-headers;mlir-libraries;mlir-tblgen;"
  projects=("${projects[@]/mlir}")
  if [ "$mlir_python_bindings" == "ON" ]; then
    echo "- including MLIR Python bindings"
    llvm_components+="MLIRPythonModules;mlir-python-sources;"
  fi
fi
if [ -z "${llvm_projects##*lld;*}" ]; then
  echo "- including LLD components"
  llvm_enable_zlib=ON # certain system libraries are compressed with ELFCOMPRESS_ZLIB, requiring zlib support for lld
  llvm_components+="lld;"
  projects=("${projects[@]/lld}")
fi
echo "- including general tools and components"
llvm_components+="cmake-exports;llvm-headers;llvm-libraries;"
llvm_components+="llvm-config;llc;llvm-ar;llvm-as;llvm-nm;llvm-symbolizer;llvm-profdata;llvm-cov;"
llvm_components+="FileCheck;count;not;"

if [ "$(echo ${projects[*]} | xargs)" != "" ]; then
  echo "- including additional project(s) "$(echo "${projects[*]}" | xargs | tr ' ' ',')
  unset llvm_components
  install_target=install
else 
  install_target=install-distribution-stripped
  if [ -n "$mlir_python_bindings" ]; then
    # Cherry-pick the necessary commit to have a distribution target
    # for the mlir-python-sources; to be removed after we update to LLVM 17.
    echo "Cherry-picking commit 9494bd84df3c5b496fc087285af9ff40d7859b6a"
    git cherry-pick --no-commit 9494bd84df3c5b496fc087285af9ff40d7859b6a
    if [ ! 0 -eq $? ]; then
      echo "Cherry-pick failed."
      if $(git rev-parse --is-shallow-repository); then
        echo "Unshallow the repository and try again."
        (return 0 2>/dev/null) && return 1 || exit 1
      fi
    fi
  fi
fi

# A hack, since otherwise the build can fail due to line endings in the LLVM script:
cat "$LLVM_SOURCE/llvm/cmake/config.guess" | tr -d '\r' > ~config.guess
cat ~config.guess > "$LLVM_SOURCE/llvm/cmake/config.guess" && rm -rf ~config.guess

# Generate CMake files.
cmake_args=" \
  -DLLVM_DEFAULT_TARGET_TRIPLE='"$(bash $LLVM_SOURCE/llvm/cmake/config.guess)"' \
  -DCMAKE_BUILD_TYPE=$build_configuration \
  -DCMAKE_INSTALL_PREFIX='"$LLVM_INSTALL_PREFIX"' \
  -DLLVM_ENABLE_PROJECTS='"${llvm_projects%;}"' \
  -DLLVM_ENABLE_RUNTIMES='"${llvm_runtimes%;}"' \
  -DLLVM_DISTRIBUTION_COMPONENTS='"${llvm_components%;}"' \
  -DLLVM_ENABLE_ZLIB=${llvm_enable_zlib:-OFF} \
  -DZLIB_ROOT='"$ZLIB_INSTALL_PREFIX"' \
  -DPython3_EXECUTABLE='"$Python3_EXECUTABLE"' \
  -DMLIR_ENABLE_BINDINGS_PYTHON=$mlir_python_bindings \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS='-w'"

if [ -z "$LLVM_CMAKE_CACHE" ]; then 
  LLVM_CMAKE_CACHE=`find "$this_file_dir/.." -path '*/cmake/caches/*' -name LLVM.cmake`
fi
if [ -f "$LLVM_CMAKE_CACHE" ]; then 
  echo "Using CMake cache in $LLVM_CMAKE_CACHE."
  cp "$LLVM_CMAKE_CACHE" custom_toolchain.cmake
  cmake_cache='-C custom_toolchain.cmake'
  # Note on combining a CMake cache with command line definitions:
  # If a set(... CACHE ...) call in the -C file does not use FORCE, 
  # the command line define takes precedence regardless of order.
  # If set(... CACHE ... FORCE) is used, the order of definition 
  # matters and the last defined value is used.
  # See https://cmake.org/cmake/help/latest/manual/cmake.1.html.
else
  echo "No CMake file found to populate the initial cache with. Set LLVM_CMAKE_CACHE to define one."
fi

if $verbose; then
  cmake -G Ninja $LLVM_SOURCE/llvm $cmake_args $cmake_cache
else
  cmake -G Ninja $LLVM_SOURCE/llvm $cmake_args $cmake_cache \
    2> logs/cmake_error.txt 1> logs/cmake_output.txt
fi

# Build and install the defined distribution.
echo "Building LLVM with configuration $build_configuration..."
if $verbose; then
  ninja $install_target
  status=$?
else
  echo "The progress of the build is being logged to `pwd`/logs/ninja_output.txt."
  ninja $install_target 2> logs/ninja_error.txt 1> logs/ninja_output.txt
  status=$?
fi

if [ "$status" = "" ] || [ ! "$status" -eq "0" ]; then
  echo "Failed to build compiler components. Please check the files in the `pwd`/logs directory."
  cd "$working_dir" && (return 0 2>/dev/null) && return 1 || exit 1
else
  cp bin/llvm-lit "$LLVM_INSTALL_PREFIX/bin/"
fi

# Build and install runtimes using the newly built toolchain.
if [ -n "$llvm_runtimes" ]; then
  echo "Building runtime components..."
  ninja runtimes
  ninja install-runtimes
  status=$?
  if [ "$status" = "" ] || [ ! "$status" -eq "0" ]; then
    echo "Failed to build runtime components. Please check the files in the `pwd`/logs directory."
    cd "$working_dir" && (return 0 2>/dev/null) && return 1 || exit 1
  else
    # We can use a default config file to set specific clang configurations.
    # See https://clang.llvm.org/docs/UsersManual.html#configuration-files
    clang_config_file="$LLVM_INSTALL_PREFIX/bin/clang++.cfg"
    echo '-L"'$LLVM_INSTALL_PREFIX/lib'"' > "$clang_config_file"
    echo '-Wl,-rpath,"'$LLVM_INSTALL_PREFIX/lib'"' >> "$clang_config_file"
    target_specific_libs=`ls -d "$LLVM_INSTALL_PREFIX/lib"/*linux*`
    for libdir in $target_specific_libs; do
      echo '-L"'$libdir'"' >> "$clang_config_file"
      echo '-Wl,-rpath,"'$libdir'"' >> "$clang_config_file"
    done
    echo "Added default configuration $clang_config_file."
  fi
fi

cd "$working_dir" && echo "Installed llvm build in directory: $LLVM_INSTALL_PREFIX"
