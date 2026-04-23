# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import os
import subprocess
import sys

# This script uses the library search logic embedded in the cuda-quantum wheels
# to help facilitate starting `cudaq-qpud` in a wheels-only environment.

if "CUDAQ_DYNLIBS" in os.environ:
    library_paths = os.environ["CUDAQ_DYNLIBS"]
    individual_paths = library_paths.split(":")
    lib_dirs = {os.path.dirname(path) for path in individual_paths}
    ld_library_path = ":".join(lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = ld_library_path + ":" + os.environ.get(
        "LD_LIBRARY_PATH", "")

cudaq_qpud = os.path.dirname(cudaq.__file__) + "/../bin/cudaq-qpud"
result = subprocess.run([cudaq_qpud] + sys.argv[1:], text=True)
