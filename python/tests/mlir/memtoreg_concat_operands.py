# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP %s | FileCheck %s

import cudaq
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


@cudaq.kernel
def runtime_indexed_flips():
    q = cudaq.qvector(2)
    for i in range(10):
        x(q[i % 2])
    mz(q)


def run_passes(kernel, pipeline):
    module = cudaq_runtime.cloneModule(kernel.qkeModule)
    pm = PassManager.parse(f"builtin.module({pipeline})",
                           context=module.context)
    cudaq_runtime.runPassManager(pm, module)
    return str(module)


def test_concat_operands_stay_in_memory():
    # `factor-quantum-alloc` splits the register into two single qubit
    # allocations and joins them with a `quake.concat` for the run-time indexed
    # access. The loop writes the qubits through the concat, so `memtoreg` must
    # not promote the allocations to wires: a wire taken before the loop would
    # never see the writes made in the loop.
    print("CONCAT_OPERANDS")
    print(
        run_passes(
            runtime_indexed_flips,
            "func.func(factor-quantum-alloc,canonicalize,memtoreg{quantum=1})"))


# CHECK-LABEL: CONCAT_OPERANDS
# CHECK:       func.func @__nvqpp__mlirgen__runtime_indexed_flips
# CHECK:         %[[A:.*]] = quake.alloca !quake.ref
# CHECK-NEXT:    %[[B:.*]] = quake.alloca !quake.ref
# CHECK-NEXT:    %[[Q:.*]] = quake.concat %[[A]], %[[B]] : (!quake.ref, !quake.ref) -> !quake.veq<2>
# CHECK-NOT:     quake.unwrap %[[A]]
# CHECK-NOT:     quake.unwrap %[[B]]
# CHECK:         cc.loop while
# CHECK:           %[[R:.*]] = quake.extract_ref %[[Q]][%{{.*}}]
# CHECK:           %[[W:.*]] = quake.unwrap %[[R]]
# CHECK:           %[[X:.*]] = quake.x %[[W]]
# CHECK:           quake.wrap %[[X]] to %[[R]]
# CHECK:         quake.mz %[[Q]]
# CHECK:         %[[WA:.*]] = quake.unwrap %[[A]]
# CHECK:         quake.sink %[[WA]]
# CHECK:         %[[WB:.*]] = quake.unwrap %[[B]]
# CHECK:         quake.sink %[[WB]]
