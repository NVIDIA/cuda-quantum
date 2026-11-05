# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import importlib.util

import cudaq
import math

# Rotation synthesis is randomized; seed it so that the counts are reproducible.
cudaq.set_random_seed(42)


# [Begin Kernel]
@cudaq.kernel
def trotter_steps(q: cudaq.qview):
    # Four Trotter steps of an 8-qubit Ising-like chain.
    for step in range(4):
        scale = step + 1
        for i in range(8):
            rz(math.pi / (8 * scale), q[i])
        for left in range(0, 7, 2):
            x.ctrl(q[left], q[left + 1])
            rz(math.pi / (16 * scale), q[left + 1])
            x.ctrl(q[left], q[left + 1])
        for left in range(1, 7, 2):
            x.ctrl(q[left], q[left + 1])
            rz(math.pi / (12 * scale), q[left + 1])
            x.ctrl(q[left], q[left + 1])


@cudaq.kernel
def trotter():
    q = cudaq.qvector(8)
    trotter_steps(q)
    mz(q)


# [End Kernel]

# [Begin IQM]
cudaq.set_target("iqm", emulate=True)

results = cudaq.estimate(trotter)
# 468 gates on 8 qubits: 404 `phased_rx`, 56 `cz` and 8 `mz`.
print(results.resources)
# [End IQM]

# [Begin CliffordT]
CLIFFORD_T_TARGET = "compiler-bench-ftqc-clifford-t"
cudaq.set_target(CLIFFORD_T_TARGET)

results = cudaq.estimate(trotter)
# 3714 gates on 8 qubits: 1507 `h`, 674 `s`, 1492 `t`, 8 `x`, 11 `z`,
# 14 `cx` and 8 `mz`.
print(results.resources)
# [End CliffordT]

# [Begin OptimizationLevel]
cudaq.set_target(CLIFFORD_T_TARGET)

# T count per level: 5978 (level 0), 5676 (level 1) and 1492 (levels 2 and 3).
for level in (0, 1, 2, 3):
    resources = cudaq.estimate(trotter, optimization_level=level).resources
    print(f'optimization_level = {level}:\n{resources}\n')
# [End OptimizationLevel]

# The cudaq-logical package is optional: skip this part if it is not installed.
if importlib.util.find_spec("cudaq.logical") is not None:
    # [Begin Logical]
    import cudaq.logical as cql

    cudaq.set_target(cql.targets.clifford_t_target(precision=1e-10))
    # level 0:  5978 T
    # level 1:  5676 T
    # level 2:  1492 T
    for level in range(3):
        estimate = cudaq.estimate(trotter, optimization_level=level)
        print(f"level {level}: {estimate.resources.count('t'):>5} T")
    # [End Logical]

# [Begin GetCompileTarget]
from cudaq._experimental import get_compile_target

cudaq.set_target(CLIFFORD_T_TARGET)
clifford_t_target = get_compile_target()
# CompileTarget(stages=[CompilationStage(ir_output_profile=QuakeUnmappedProfile(
#     basis_gates=['h', 's', 't', 'x', 'y', 'z', 'swap', 'x(1)', 'y(1)', 'z(1)'],
#     approximation_tolerance=1e-10, cost_model=CostModel.t_count, ...),
#     lowering=inferred)], ...)
print(clifford_t_target)
# [End GetCompileTarget]

# [Begin ModifyTarget]
new_basis_gates = ['x(1)', 'h', 'rz']

stage = clifford_t_target.stages[0]
profile = stage.ir_output_profile
profile.basis_gates = new_basis_gates
stage.ir_output_profile = profile
clifford_t_target.stages = [stage]
# Same as before, but with `basis_gates=['x(1)', 'h', 'rz']`.
print(clifford_t_target)

# Pass a named target and a compile target: kernels run on the named target,
# but are compiled against the given compile target.
cudaq.set_target("qpp-cpu", compile_target=clifford_t_target)
# Gate counts per level, in addition to the 8 `mz` measurements:
#   level 0: 56 `cx` and 60 `rz`
#   level 1: 56 `cx` and 57 `rz`
#   level 2: 14 `cx` and 15 `rz`
for level in range(3):
    resources = cudaq.estimate(trotter, optimization_level=level).resources
    print(resources)
# [End ModifyTarget]

# [Begin CustomTarget]
from cudaq._experimental import CompileTarget, QuakeUnmappedProfile, CostModel

nisq_gates = ["h", "rx", "rz", "z(1)"]
nisq_profile = QuakeUnmappedProfile(basis_gates=nisq_gates,
                                    cost_model=CostModel.two_qubit_gate_count)
nisq_target = CompileTarget(ir_output_profile=nisq_profile)
cudaq.set_target("qpp-cpu", compile_target=nisq_target)
# Gate counts per level, in addition to the 8 `mz` measurements:
#   level 0: 56 `cz`, 112 `h` and 60 `rz`
#   level 1: 56 `cz`, 112 `h` and 57 `rz`
#   level 2: 56 `cz`, 112 `h` and 15 `rz`
for level in range(3):
    resources = cudaq.estimate(trotter, optimization_level=level).resources
    print(resources)
# [End CustomTarget]

# [Begin CustomPipeline]
from cudaq._experimental import CustomPipelineLowering

# Just decompose the gates and do nothing else.
custom_pipeline = "decomposition{basis=" + ",".join(nisq_gates) + "}"
custom_target = CompileTarget(custom_pipeline=CustomPipelineLowering(
    mid_level=custom_pipeline))
cudaq.set_target("qpp-cpu", compile_target=custom_target)

resources = cudaq.estimate(trotter).resources
# 236 gates: 56 `cz`, 112 `h`, 60 `rz` and 8 `mz`.
print(resources)
# [End CustomPipeline]

cudaq.reset_target()
