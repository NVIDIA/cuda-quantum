# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import concurrent.futures

import cudaq
import pytest
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


@pytest.fixture(autouse=True)
def run_and_clear_registries():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


@cudaq.kernel
def threaded_compile_kernel(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits)
    mz(qubits)


def test_concurrent_compile_on_python_mlir_context():
    threaded_compile_kernel.compile()

    def compile_once(_):
        args, module = threaded_compile_kernel.prepare_call(4)
        cudaq_runtime.marshal_and_retain_module(
            threaded_compile_kernel.uniqName, module, True, *args)

    for _ in range(4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(compile_once, range(4)))
