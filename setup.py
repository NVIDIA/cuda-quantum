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
    print("Trying to install required module: skbuild")
    os.system(f"{sys.executable} -m pip install scikit-build")
import skbuild
import setuptools

__version__ = "0.3.0"

# The setup script gets called twice when installing from source
# with `pip install .` . Once for the `egg_info` subcommand and another
# for `install`. We will only install the missing dependencies once, for
# the `egg_info` subcommand (arbitrary choice).
if (sys.argv[1] == 'egg_info'):
    script_path = os.getcwd() + "/scripts/install_wheel_dependencies.sh"
    os.system(f"bash {script_path}")

# FIXME: support installation without --user flag
# GitHub issue: https://github.com/NVIDIA/cuda-quantum/issues/125 
# Linux machines default to dist-packages unless the `--user` flag is provided to
# the pip install. We hard-code everything to site-packages in the meantime and require the
# user to install with `--user`.
cmake_install_dir = f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages/cudaq"

skbuild.setup(
    name="cuda-quantum",
    version=__version__,
    package_dir={"cudaq": "python/cudaq"},
    packages=setuptools.find_packages(where="python", include=["cudaq"]),
    zip_safe=False,
    python_requires=">=3.8",
    cmake_install_dir=cmake_install_dir,
    cmake_args=[
        "-DCUDAQ_ENABLE_PYTHON=TRUE", "-DBLAS_LIBRARIES=/usr/lib64/libblas.a",
        "-DCMAKE_INSTALL_LIBDIR=lib", "-DCUDAQ_DISABLE_CPP_FRONTEND=ON",
        "-DCUDAQ_BUILD_TESTS=FALSE", "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
        "-DCUSTATEVEC_ROOT={}".format(os.environ["CUSTATEVEC_ROOT"])
        if "CUSTATEVEC_ROOT" in os.environ else "",
        "-DLLVM_DIR={}".format(os.environ["LLVM_DIR"] if "LLVM_DIR" in
                               os.environ else "/opt/llvm"),
        "-DOPENSSL_USE_STATIC_LIBS=TRUE",
        "-DCMAKE_EXE_LINKER_FLAGS='-static-libgcc -static-libstdc++'",
        "-DCMAKE_SHARED_LINKER_FLAGS='-static-libgcc -static-libstdc++'",
        "-DOPENSSL_ROOT_DIR=/usr/local/ssl",
        "-DCUDAQ_CPR_INSTALL={}".format(os.environ["CPR_DIR"] if "CPR_DIR" in
                                        os.environ else "/cpr/install"),
        "-DCUDAQ_BUILD_RELOCATABLE_PACKAGE=TRUE"
    ],
    setup_requires=["numpy", "pytest", "scikit-build", "setuptools"])
