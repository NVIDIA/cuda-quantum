# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import os, os.path
from ._packages import *

if not "CUDAQ_DYNLIBS" in os.environ:
    try:
        cublas_libs = get_library_path("nvidia-cublas-cu11")
        cublas_path = os.path.join(cublas_libs, "libcublas.so.11")
        cublasLt_path = os.path.join(cublas_libs, "libcublasLt.so.11")

        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ[
            "CUDAQ_DYNLIBS"] = f"{cublasLt_path}:{cublas_path}:{custatevec_path}:{cutensornet_path}"
    except:
        pass

from ._pycudaq import *
from .domains import chemistry

initKwargs = {'target': 'default'}

if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]

if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]

initialize_cudaq(**initKwargs)
