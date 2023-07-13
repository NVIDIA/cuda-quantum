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
    print("Installing required package scikit-build.")
    os.system(f"{sys.executable} -m pip install scikit-build")
import skbuild
import setuptools

__version__ = os.getenv("CUDA_QUANTUM_VERSION")

# The `setup.py` script gets called twice when installing from source
# with `pip install .` . Once for the `egg_info` subcommand and another
# for `install`. We will only install the missing dependencies once.
if (sys.argv[1] != 'egg_info'):
    script_path = os.getcwd() + "/scripts/install_wheel_dependencies.sh"
    # FIXME: this doesn't fail if the install fails
    os.system(f"bash {script_path}")

# FIXME: support installation without --user flag
# GitHub issue: https://github.com/NVIDIA/cuda-quantum/issues/125
# Linux machines default to dist-packages unless the `--user` flag is provided to
# the pip install. We hard-code everything to site-packages in the meantime and require the
# user to install with `--user`.
cmake_install_dir = f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages/cudaq"
packages=setuptools.find_packages(where="python", include=["cudaq"])

skbuild.setup(
    name="cuda-quantum",
    version=__version__,
    package_dir={"cudaq": "python/cudaq"},
    packages=packages,
    cmake_with_sdist=True, # we use cmake to pull some third party libraries 
    python_requires=">=3.8",
    #data_files=[ ("", "_pycudaq.*"), ],
    cmake_install_dir=cmake_install_dir,
    cmake_minimum_required_version="3.26",
    cmake_args=[
        "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF", # FIXME
        "-DCUDAQ_ENABLE_PYTHON=TRUE",
        "-DCUDAQ_DISABLE_CPP_FRONTEND=TRUE",
        "-DCUDAQ_BUILD_TESTS=FALSE",
        "-DCUDAQ_BUILD_SELFCONTAINED={}".format(os.environ["CUDAQ_BUILD_SELFCONTAINED"])
            if "CUDAQ_BUILD_SELFCONTAINED" in os.environ else "",
        "-DCUSTATEVEC_ROOT={}".format(os.environ["CUSTATEVEC_ROOT"])
            if "CUSTATEVEC_ROOT" in os.environ else "",
        "-DLLVM_DIR={}/lib/cmake/llvm".format(os.environ["LLVM_INSTALL_PREFIX"])
            if "LLVM_INSTALL_PREFIX" in os.environ else "",
        "-DOPENSSL_ROOT_DIR={}".format(os.environ["OPENSSL_ROOT_DIR"])
            if "OPENSSL_ROOT_DIR" in os.environ else "",
        "-DBLAS_LIBRARIES={}".format(os.environ["BLAS_LIBRARIES"])
            if "BLAS_LIBRARIES" in os.environ else "",
    ])
