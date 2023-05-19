# ******************************************************************************
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the MIT License which accompanies this distribution.
# ******************************************************************************
import os
import sys
try:
    import skbuild
except ImportError:
    print("Trying to install required module: skbuild")
    os.system("{} -m pip install scikit-build --user".format(sys.executable))
import skbuild
import setuptools
skbuild.setup(
    name="cudaq",
    version="0.0.3",
    package_dir={"": "python"},
    packages=setuptools.find_packages(where="python", include=["*"]),
    zip_safe=False,
    python_requires=">=3.6",
    # This ensures that the python package is in the PYTHONPATH
    cmake_install_dir="lib/python{}.{}/site-packages/cuda-quantum".format(sys.version_info[0],sys.version_info[1]),
    cmake_args=["-DCUDAQ_ENABLE_PYTHON=TRUE",
                "-DBLAS_LIBRARIES=/usr/lib64/libblas.a",
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DCUSTATEVEC_ROOT={}".format(
                    os.environ["CUSTATEVEC_ROOT"]) if "CUSTATEVEC_ROOT" in os.environ else "",
                "-DLLVM_DIR={}".format(os.environ["LLVM_DIR"]) if "LLVM_DIR" in os.environ else "",
                "-DOPENSSL_USE_STATIC_LIBS=TRUE", 
                "-DCMAKE_EXE_LINKER_FLAGS='-static-libgcc -static-libstdc++'", 
                "-DCMAKE_SHARED_LINKER_FLAGS='-static-libgcc -static-libstdc++'",
                "-DOPENSSL_ROOT_DIR=/usr/local/ssl",
                "-DCUDAQ_CPR_INSTALL={}".format(
                    os.environ["CUDAQ_CPR_INSTALL"]), 
                "-DZLIB_ROOT={}".format(
                    os.environ["CUDAQ_CPR_INSTALL"]), 
                "-DZLIB_USE_STATIC_LIBS=TRUE", 
                "-DCUDAQ_BUILD_RELOCATABLE_PACKAGE=TRUE"])