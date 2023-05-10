# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import sys

try:
    import skbuild
except ImportError:
    print("Trying to install required module: scikit-build")
    os.system(f"{sys.executable} -m pip install scikit-build")

# try:
#     import pybind11
# except:
#     # NOTE: We may have an issue with our git installation not containing a 
#     # pybind11Config.cmake file. Hopefully this sorts it out, but a conda
#     # installation may be required otherwise.
#     # See: https://github.com/pybind/pybind11/issues/1379
#     # Working solution has been to copy the .cmake file generated with a pip
#     # install into our `tpls/pybind11/tools` directory.

import skbuild
import setuptools

__version__ = "0.0.1"

skbuild.setup(
    name="cuda-quantum",
    version=__version__,
    url="https://github.com/NVIDIA/cuda-quantum",
    long_description="",
    package_dir={"": "/cuda-quantum/python"},
    packages=setuptools.find_packages(where="/cuda-quantum/python",
                                      include=["*"]),
    zip_safe=False,

    # TODO: Will extend for further python versions after a proof of concept.
    # We will likely need to test each version that we support.
    python_requires=">=3.10",

    # TODO: Replace all of the hard-coded paths

    # FIXME: Have to find the correct path to use here.
    # This ensures that the python package is in the PYTHONPATH
    cmake_install_dir=
    f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/cuda-quantum",
    # TODO: Add cmake flag to remove the need to build the nvq++ frontend here
    cmake_args=[
        "-DCUDAQ_ENABLE_PYTHON=TRUE",
        "-DLLVM_DIR=/root/../opt/llvm/clang-16/lib/cmake/llvm"
        "-DCUSTATEVEC_ROOT=/opt/nvidia/cuquantum",
        "-DCMAKE_EXE_LINKER_FLAGS_INIT=$/opt/llvm/bin/ld.lld"
        "-DCMAKE_MODULE_LINKER_FLAGS_INIT=$/opt/llvm/bin/ld.lld"
        "-DCMAKE_SHARED_LINKER_FLAGS_INIT=$/opt/llvm/bin/ld.lld",
        "-DOPENSSL_ROOT_DIR=/usr/local/ssl",
        "-DCUDAQ_CPR_INSTALL=/lib/x86_64-linux-gnu/libz.so.1.2.11",
        "-DOPENSSL_USE_STATIC_LIBS=TRUE",
        "-DCUDAQ_BUILD_RELOCATABLE_PACKAGE=TRUE",

        # First attempt at disabling the front-end build to minimize
        # dependencies of the pip wheel.
        "-CUDAQ_DISABLE_FRONTEND=TRUE",

        # NOTE: Not yet sure what/where to put the build output.
        "-DCMAKE_INSTALL_LIBDIR=lib",

        # Hopefully linking to my hard-coded build of zlib properly.
        "-DZLIB_ROOT=/lib/x86_64-linux-gnu/libz.so.1.2.11",
        "-DZLIB_USE_STATIC_LIBS=TRUE"

        # NOTE: I replaced these with similar but different commands from
        # the shell build script. May have to add them back.
        "-DCMAKE_EXE_LINKER_FLAGS='-static-libgcc -static-libstdc++'",
        "-DCMAKE_SHARED_LINKER_FLAGS='-static-libgcc -static-libstdc++'",

    ],
    setup_requires=["numpy", "pytest", "scikit-build"])
