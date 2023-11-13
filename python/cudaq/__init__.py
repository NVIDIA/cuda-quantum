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
        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ["CUDAQ_DYNLIBS"] = f"{custatevec_path}:{cutensornet_path}"
    except:
        import importlib.util
        if not importlib.util.find_spec("cuda-quantum") is None:
            print("Could not find a suitable cuQuantum Python package.")
        pass

from ._pycudaq import *
from .domains import chemistry

initKwargs = {}
if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]

if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]

initialize_cudaq(**initKwargs)
