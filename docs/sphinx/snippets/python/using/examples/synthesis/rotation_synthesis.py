# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Target]
import cudaq

# Rewrite every rotation in the kernel into Clifford+T. `epsilon` is the
# per-rotation approximation tolerance.
cudaq.set_target("compiler-bench-ftqc-clifford-t", epsilon="1e-10")


@cudaq.kernel
def circuit():
    q = cudaq.qvector(2)
    h(q[0])
    rz(0.6, q[0])
    ry(0.25, q[1])
    x.ctrl(q[0], q[1])
    mz(q)


operations = cudaq.estimate_resources(circuit).to_dict()

# Only Clifford+T operations are left. The rotations are gone.
CLIFFORD_T = {
    "h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "cy", "cz", "swap", "mx",
    "my", "mz"
}

print(f"only Clifford+T: {set(operations) <= CLIFFORD_T}")
print(f"T count:         {operations.get('t', 0) + operations.get('tdg', 0)}")

cudaq.reset_target()
# [End Target]

# [Begin Single]
from cudaq import synth

theta, epsilon = 0.6, 1e-10

# `seed` is optional. Passing one makes the result reproducible. Without it,
# repeated calls may return different sequences of the same length.
seq = synth.gridsynth(theta, epsilon, seed=1234)

print(f"T count: {seq.t_count}")
print(f"gates:   {str(seq)[:32]}...")

# The achieved error never exceeds the epsilon that was asked for. This is
# computed exactly, independent of any simulator's precision.
print(f"error:   {synth.rz_error(theta, seq):.3e}")
assert synth.rz_error(theta, seq) <= epsilon
# [End Single]

# [Begin Kernel]
# `to_kernel` builds a kernel taking a single qubit, for use with `apply_call`.
# Sandwiching the rotation between two Hadamards turns the phase it applies
# into a measurable population, so the counts below depend on theta.
kernel = cudaq.make_kernel()
qubit = kernel.qalloc()
kernel.h(qubit)
kernel.apply_call(seq.to_kernel(), qubit)
kernel.h(qubit)
kernel.mz(qubit)

# The same circuit built with an exact `rz`, for comparison.
exact = cudaq.make_kernel()
exact_qubit = exact.qalloc()
exact.h(exact_qubit)
exact.rz(theta, exact_qubit)
exact.h(exact_qubit)
exact.mz(exact_qubit)

cudaq.set_random_seed(13)
print(f"synthesized: {cudaq.sample(kernel, shots_count=1000)}")
cudaq.set_random_seed(13)
print(f"exact rz:    {cudaq.sample(exact, shots_count=1000)}")
# [End Kernel]
