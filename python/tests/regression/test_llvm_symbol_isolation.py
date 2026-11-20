# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q must not interpose on unrelated extensions' LLVM/MLIR symbols."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux",
                                reason="ELF symbol scope")


def run_child(tmp_path, source):
    script = tmp_path / "symbol_isolation.py"
    script.write_text(textwrap.dedent(source))
    env = dict(os.environ,
               NUMBA_CACHE_DIR=str(tmp_path / "numba-cache"),
               CUDAQ_DEFAULT_SIMULATOR="qpp-cpu")
    result = subprocess.run([sys.executable, str(script)],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_llvm_symbols_remain_local_after_kernel_execution(tmp_path):
    run_child(
        tmp_path, """
        import ctypes
        process = ctypes.CDLL(None)
        symbols = (
            "_ZN4llvm2cl6Option11addArgumentEv",
            "_ZTVN4llvm21BranchProbabilityInfo20BasicBlockCallbackVHE",
            "_ZN4mlir11MLIRContextC1ENS0_9ThreadingE",
        )
        def check_scope():
            for symbol in symbols:
                assert not hasattr(process, symbol), symbol

        check_scope()
        import cudaq
        check_scope()

        @cudaq.kernel
        def flip():
            q = cudaq.qubit()
            x(q)
            mz(q)

        builder = cudaq.make_kernel()
        q = builder.qalloc()
        builder.x(q)
        builder.mz(q)
        targets = ["qpp-cpu", "density-matrix-cpu", "stim"]
        if cudaq.num_available_gpus() and cudaq.has_target("nvidia"):
            targets.append("nvidia")
        for target in targets:
            cudaq.set_target(target)
            for kernel in (flip, builder):
                assert cudaq.sample(kernel, shots_count=10)["1"] == 10
            check_scope()
    """)


@pytest.mark.parametrize("cudaq_first", [True, False])
@pytest.mark.skipif(importlib.util.find_spec("numba") is None,
                    reason="Numba interoperability requires numba")
def test_numba_jit_import_order(tmp_path, cudaq_first):
    imports = "import cudaq\nimport numba" if cudaq_first else "import numba\nimport cudaq"
    run_child(
        tmp_path, imports + "\n" + textwrap.dedent("""
        import numpy as np

        @numba.njit
        def f(s):
            return np.sum(s > 0.5)

        assert f(np.array([0.0, 0.25, 0.75, 1.0])) == 2
        cudaq.set_target("qpp-cpu")
        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.x(q)
        assert cudaq.sample(kernel, shots_count=10)["1"] == 10
    """))


@pytest.mark.parametrize("cudaq_first", [True, False])
@pytest.mark.skipif(importlib.util.find_spec("triton") is None,
                    reason="Triton interoperability requires triton")
def test_triton_import_order(tmp_path, cudaq_first):
    imports = "import cudaq\nimport triton" if cudaq_first else "import triton\nimport cudaq"
    run_child(
        tmp_path, imports + "\n" + textwrap.dedent("""
        cudaq.set_target("qpp-cpu")
        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.x(q)
        assert cudaq.sample(kernel, shots_count=10)["1"] == 10
    """))
