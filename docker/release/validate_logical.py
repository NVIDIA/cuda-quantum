# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exercise the `cudaq.logical` installation as the release image's CUDA-Q user."""

from importlib.metadata import distribution
import os
from pathlib import Path
import subprocess

import cudaq.logical as ql

prefix = Path(os.environ["CUDA_QUANTUM_PATH"]).resolve()
metadata = distribution("cudaq-logical")
assert Path(ql.__file__).resolve().is_relative_to(prefix / "cudaq/logical")
assert Path(metadata.locate_file("")).resolve() == prefix
assert ql.__version__ == metadata.version

# Simple program to exercise the `cudaq.logical` installation.
# TODO: Use `examples` directory once location is finalized.


@ql.program
def bell() -> tuple[bool, bool]:
    q = ql.allocate(2, state=ql.types.zero)
    q[0] = ql.h(q[0])
    q[0], q[1] = ql.cx(q[0], q[1])
    return ql.measure_z(q[0]), ql.measure_z(q[1])


build = ql.compile(bell)
estimate = ql.estimate(build, tier=ql.estimate.Tier.LOGICAL)
assert estimate.logical_qubits_peak == 2
assert estimate.actions == {"qlx_standard_h": 1, "qlx_standard_cx": 1}

subprocess.run(["qlx-opt"], input="module {}\n", text=True, check=True)
subprocess.run(["qlx-translate", "--help"],
               check=True,
               stdout=subprocess.DEVNULL)
print(f"CUDA-Q Logical {ql.__version__}: image validation passed")
