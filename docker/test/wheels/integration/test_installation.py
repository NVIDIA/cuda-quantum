# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Smoke-test installed candidate wheels, from outside the source/build trees.

Run once with core only, once with --logical, and once with --logical --frontend.
Environment creation and pip installation stay with the caller; no import hooks
or source-path injection are used here.
"""
import argparse
from importlib.metadata import distribution, distributions
from pathlib import Path
import re
import sys


def check_ownership(logical, frontend):
    """Core owns its files and shared libraries, including after wheel repair."""
    core_files = {str(p) for p in distribution("cudaq-core").files}
    core_python_files = {
        p.removeprefix("cudaq_core/")
        for p in core_files
        if p.startswith("cudaq_core/cudaq/")
    }

    def libraries(files):
        # Repair can add a hash to a library name; that is still a duplicate.
        return {
            re.sub(r"-[0-9a-f]{8,}(?=\.)", "",
                   Path(p).name)
            for p in files
            if re.search(r"\.so(?:\.\d+)*$|\.dylib$", p)
        }

    dependents = ["cudaq-logical"] if logical else []
    if frontend:
        dependents.extend(d.metadata["Name"]
                          for d in distributions()
                          if d.metadata["Name"].startswith("cuda-quantum-"))
    for name in dependents:
        files = {str(p) for p in distribution(name).files}
        assert not core_files & files, (name, core_files & files)
        # Different install roots must not provide the same cudaq package file.
        overlap = core_python_files & files
        assert not overlap, (name, overlap)
        duplicates = libraries(core_files) & libraries(files)
        assert not duplicates, (name, duplicates)


def check_core(frontend):
    import cudaq
    from cudaq.mlir import ir, passmanager
    from cudaq.mlir.dialects import quake

    with ir.Context() as context:
        quake.register_dialect(context=context)
        module = ir.Module.parse("module { func.func @identity() { return } }")
        passmanager.PassManager.parse("builtin.module(canonicalize,cse)").run(
            module.operation)
        assert module.operation.verify()

    if not frontend:
        assert cudaq.__spec__.origin is None, "Core must not own cudaq/__init__.py"
        assert "cudaq.mlir._mlir_libs._quakeDialects" not in sys.modules
        assert "cudaq.kernel.kernel_decorator" not in sys.modules

    # An accidental private MLIR copy can import successfully yet corrupt types
    # passed between extensions. Check the installed process, not build artifacts.
    maps = Path("/proc/self/maps")
    if maps.exists():
        images = {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if "libcudaqMLIR" in line and "libcudaqMLIRCAPI" not in line
        }
        assert len(images) == 1, images
        if not frontend:
            assert not any("/libcudaq.so" in line or "/libnvqir" in line
                           for line in maps.read_text().splitlines())


def check_logical():
    import cudaq.logical as logical
    from cudaq.core.backends import EstimateResult
    from cudaq.util import trace

    @logical.program
    def readout() -> bool:
        return logical.measure_z(logical.prepare_zero())

    backend = trace.ChromeBackend()
    trace.set_backend(backend)
    try:
        result = logical.targets.TerminalBackend().estimate(
            logical.compile(readout), tier=logical.estimate.Tier.LOGICAL)
    finally:
        trace.reset_backend()
    assert isinstance(result, EstimateResult)
    assert logical.estimate.LogicalProfile.from_annotations(
        result.annotations).total_operations == 2
    assert "cudaq.estimate.LOGICAL" in {
        event["name"] for event in backend.to_dict()["traceEvents"]
    }


def check_frontend():
    import cudaq
    from cudaq.core import backends
    from cudaq._experimental import CompileTarget, CustomTarget
    from cudaq.mlir._mlir_libs import _backends, _quakeDialects

    assert CompileTarget is backends.CompileTarget
    assert CustomTarget is backends.CustomTarget
    assert _quakeDialects.cudaq_runtime.trace is _backends.trace
    cudaq.set_target("qpp-cpu")
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.x(qubit)
    kernel.mz(qubit)
    assert cudaq.sample(kernel, shots_count=10)["1"] == 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", action="store_true")
    parser.add_argument("--frontend", action="store_true")
    args = parser.parse_args()
    check_ownership(args.logical, args.frontend)
    if args.logical:
        check_logical()
    if args.frontend:
        check_frontend()
    check_core(args.frontend)
    print("Installed wheel checks passed")
