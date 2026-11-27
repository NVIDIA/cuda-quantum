# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exercise the `cudaq-edem` installation as the release image's CUDA-Q user."""

import ctypes
from importlib.metadata import distribution
from pathlib import Path

import cudaq_edem

metadata = distribution("cudaq-edem")
assert cudaq_edem.__version__ == metadata.version

# The package carries its native libraries and CMake files for C/C++ consumers.
cmake = Path(cudaq_edem.get_cmake_dir())
assert (cmake / "cudaq-edem-config.cmake").is_file()
library_dir = cmake.parent.parent
for name in ("cudaq-edem", "cudaq-gf2"):
    ctypes.CDLL(str(next(library_dir.glob(f"lib{name}.so*"))))

projected = cudaq_edem.project_DEM(cudaq_edem.compose_chain([cudaq_edem.EDEM()
                                                            ]))
assert cudaq_edem.DEM.from_bytes(projected.to_bytes()).is_dem

# EDEM is independent of CUDA-Q, but both must be usable in one process.
import cudaq

assert cudaq.__file__ and cudaq_edem.__file__
print(f"cudaq-edem {cudaq_edem.__version__}: image validation passed")
