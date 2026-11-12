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
def inverse_qft(q: cudaq.qview):
    n = q.size()
    for i in range(n // 2):
        swap(q[i], q[n - i - 1])
    for i in range(n):
        for j in range(i):
            r1.ctrl(-math.pi / 2**(i - j), q[j], q[i])
        h(q[i])


@cudaq.kernel
def controlled_u(control: cudaq.qubit, system: cudaq.qview):
    # Controlled exp(-i theta Z0 Z1 / 2), up to a global phase.
    x.ctrl(system[0], system[1])
    rz.ctrl(0.37, control, system[1])
    x.ctrl(system[0], system[1])


@cudaq.kernel
def qpe():
    # Quantum phase estimation with 4 counting qubits, applying the
    # controlled unitary 2^k times for the k-`th` counting qubit.
    counting = cudaq.qvector(4)
    system = cudaq.qvector(2)
    x(system[0])
    h(counting)
    for k in range(4):
        for _ in range(2**k):
            controlled_u(counting[k], system)
    inverse_qft(counting)
    mz(counting)


# [End Kernel]

# [Begin IQM]
cudaq.set_target("iqm", emulate=True)

results = cudaq.estimate(qpe)
# 555 gates on 6 qubits: 473 `phased_rx`, 78 `cz` and 4 `mz`.
print(results.resources)
# [End IQM]

# [Begin CliffordT]
CLIFFORD_T_TARGET = "compiler-bench-ftqc-clifford-t"
cudaq.set_target(CLIFFORD_T_TARGET)

results = cudaq.estimate(qpe)
# 3262 gates on 6 qubits: 1313 `h`, 610 `s`, 1301 `t`, 6 `x`, 4 `z`,
# 22 `cx`, 2 `swap` and 4 `mz`.
print(results.resources)
# [End CliffordT]

# [Begin OptimizationLevel]
cudaq.set_target(CLIFFORD_T_TARGET)

# T count per level: 3897 (level 0), 3797 (level 1) and 1301 (levels 2 and 3).
for level in (0, 1, 2, 3):
    resources = cudaq.estimate(qpe, optimization_level=level).resources
    print(f'optimization_level = {level}:\n{resources}\n')
# [End OptimizationLevel]

# The cudaq-logical package is optional: skip this part if it is not installed.
if importlib.util.find_spec("cudaq.logical") is not None:
    # [Begin Logical]
    import cudaq.logical as cql

    cudaq.set_target(cql.targets.clifford_t_target(precision=1e-6))
    # 2271 T gates (versus 3897 for precision=1e-10).
    estimate = cudaq.estimate(qpe, optimization_level=0)
    print(f"{estimate.resources.count('t')} T")
    # [End Logical]

# [Begin GetCompileTarget]
from cudaq._experimental import get_compile_target

cudaq.set_target(CLIFFORD_T_TARGET)
clifford_t_target = get_compile_target()
# ```
# CompileTarget(stages=[CompilationStage(ir_output_profile=QuakeUnmappedProfile(
#     basis_gates=['h', 's', 't', 'x', 'y', 'z', 'swap', 'x(1)', 'y(1)', 'z(1)'],
#     approximation_tolerance=1e-10, cost_model=CostModel.t_count, ...),
#     lowering=inferred)], ...)
# ```
print(clifford_t_target)
# [End GetCompileTarget]

# [Begin ModifyTarget]
new_basis_gates = ['x(1)', 'h', 'x', 'rz']
new_target = clifford_t_target

stage = clifford_t_target.stages[0]
profile = stage.ir_output_profile
profile.basis_gates = new_basis_gates
stage.ir_output_profile = profile
new_target.stages = [stage]
# Same as before, but with `basis_gates=['x(1)', 'h', 'x', 'rz']`.
print(new_target)

# Pass a named target and a compile target: kernels run on the named target,
# but are compiled against the given compile target.
cudaq.set_target("qpp-cpu", compile_target=new_target)
# Gate counts per level, in addition to 8 `h`, 1 `x` and 4 `mz`:
#   level 0: 78 `cx` and 48 `rz`
#   level 1: 50 `cx` and 45 `rz`
#   level 2: 28 `cx` and 20 `rz`
for level in range(3):
    resources = cudaq.estimate(qpe, optimization_level=level).resources
    print(resources)
# [End ModifyTarget]

# [Begin CustomTarget]
from cudaq._experimental import CompileTarget, QuakeUnmappedProfile, CostModel

nisq_gates = ["h", "x", "rx", "rz", "z(1)"]
nisq_profile = QuakeUnmappedProfile(basis_gates=nisq_gates,
                                    cost_model=CostModel.two_qubit_gate_count)
nisq_target = CompileTarget(ir_output_profile=nisq_profile)
cudaq.set_target("qpp-cpu", compile_target=nisq_target)
# Gate counts per level, in addition to 1 `x` and 4 `mz`:
#   level 0: 78 `cz`, 164 `h` and 48 `rz`
#   level 1: 50 `cz`, 108 `h` and 48 `rz`
#   level 2: 28 `cz`, 64 `h` and 17 `rz`
for level in range(3):
    resources = cudaq.estimate(qpe, optimization_level=level).resources
    print(resources)
# [End CustomTarget]

# [Begin CustomPipeline]
from cudaq._experimental import CustomPipelineLowering

# Just decompose the gates and do nothing else.
custom_pipeline = "decomposition{basis=" + ",".join(nisq_gates) + "}"
custom_target = CompileTarget(custom_pipeline=CustomPipelineLowering(
    mid_level_pipeline=custom_pipeline))
cudaq.set_target("qpp-cpu", compile_target=custom_target)

resources = cudaq.estimate(qpe).resources
# 295 gates: 78 `cz`, 164 `h`, 48 `rz`, 1 `x` and 4 `mz`.
print(resources)
# [End CustomPipeline]

cudaq.reset_target()
