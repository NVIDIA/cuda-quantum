# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate the Gidney--Ekerå RSA-2048 resource envelope with CUDA-Q and CUDA-Q Logical.

The illustrative arithmetic includes QROM lookup, carry-runway addition, and
measurement unlookup. Placeholder lookup data make this a resource study, not a
factor-returning implementation. CUDA-Q Logical reports the resulting logical-resource profile;
physical-qubit, runtime, and retry-risk values are analytical projections.
"""
#%%
# Define the lookup arithmetic and RSA-2048 resource kernel.
import cudaq

from cudaq.logical.targets import estimator
from cudaq.logical.algorithms import estimate_gidney_ekera, gidney_ekera_factor
from cudaq.logical.estimate import LogicalProfile

N_BITS = 2_048
ACCUMULATOR_WIDTH = 2_086
C_SEP = 1_024
N_EXPONENT = 3_072
WINDOW = 5
LOOKUP_ADDITIONS = 164
FOLDED_LOOKUPS = N_EXPONENT * LOOKUP_ADDITIONS


@cudaq.kernel
def toffoli(c1: cudaq.qubit, c2: cudaq.qubit, target: cudaq.qubit):
    h(target)
    z.ctrl([c1, c2], target)
    h(target)


@cudaq.kernel
def maj(carry: cudaq.qubit, addend: cudaq.qubit, accumulator: cudaq.qubit):
    x.ctrl(accumulator, addend)
    x.ctrl(accumulator, carry)
    toffoli(carry, addend, accumulator)


@cudaq.kernel
def uma(carry: cudaq.qubit, addend: cudaq.qubit, accumulator: cudaq.qubit):
    toffoli(carry, addend, accumulator)
    x.ctrl(accumulator, carry)
    x.ctrl(carry, addend)


@cudaq.kernel
def lookup_addition(accumulator: cudaq.qview):
    """One c=5 RSA-2048 lookup-addition from the historical kernel."""

    address = cudaq.qvector(10)
    bus = cudaq.qvector(2086)
    runways = cudaq.qvector(3)
    ancillas = cudaq.qvector(2)
    h(address)

    for table_index in range(1023):
        toffoli(address[table_index % 10], ancillas[0], ancillas[1])
        x.ctrl(ancillas[1], bus[table_index % 2086])

    maj(runways[0], bus[0], accumulator[0])
    for offset in range(1023):
        bit = offset + 1
        maj(accumulator[bit - 1], bus[bit], accumulator[bit])
    for offset in range(1023):
        bit = 1023 - offset
        uma(accumulator[bit - 1], bus[bit], accumulator[bit])
    uma(runways[0], bus[0], accumulator[0])

    maj(runways[1], bus[1024], accumulator[1024])
    for offset in range(1023):
        bit = offset + 1025
        maj(accumulator[bit - 1], bus[bit], accumulator[bit])
    for offset in range(1023):
        bit = 2047 - offset
        uma(accumulator[bit - 1], bus[bit], accumulator[bit])
    uma(runways[1], bus[1024], accumulator[1024])

    maj(runways[2], bus[2048], accumulator[2048])
    for offset in range(37):
        bit = offset + 2049
        maj(accumulator[bit - 1], bus[bit], accumulator[bit])
    for offset in range(37):
        bit = 2085 - offset
        uma(accumulator[bit - 1], bus[bit], accumulator[bit])
    uma(runways[2], bus[2048], accumulator[2048])

    for bit in range(2086):
        mx(bus[bit])
    for fixup in range(31):
        toffoli(address[fixup % 10], ancillas[0], ancillas[1])
    for bit in range(10):
        mx(address[bit])
    for bit in range(3):
        mz(runways[bit])
    for bit in range(2):
        mz(ancillas[bit])


@cudaq.kernel
def rsa2048_resource_kernel():
    """Repeat the historical Table-3 c=5 lookup-addition workload."""

    accumulator = cudaq.qvector(2086)
    for _ in range(503808):
        lookup_addition(accumulator)
    mx(accumulator[0])


#%%
# Build the paper workload and configure CUDA-Q Logical's estimator target.
workload = gidney_ekera_factor(
    N_BITS,
    c_exp=WINDOW,
    c_mul=WINDOW,
    c_sep=C_SEP,
)

cudaq.set_target(estimator)

#%%
# Estimate the logical profile and projected physical costs.
# Expanding 503,808 lookup additions may take time.
estimate_results = cudaq.estimate(rsa2048_resource_kernel)

logical = LogicalProfile.from_annotations(estimate_results.annotations)
paper_estimate = estimate_gidney_ekera(workload, logical_profile=logical)

#%%
# Verify and report the resource estimates.
assert workload.total_toffolis == 2_632_900_608
assert logical.logical_qubits_peak == 4_187
assert sum(logical.synthesis_demand.values()) == workload.total_toffolis

print("Gidney--Ekerå @cudaq.kernel -> cudaq.estimate:")
print(f"  selected window: c={WINDOW}")
print(f"  folded lookup additions: {FOLDED_LOOKUPS:,}")
print(f"  Logical CCZ demand: {paper_estimate.toffolis:,}")
print(f"  peak logical qubits: {logical.logical_qubits_peak:,}")
print(f"  lookup cycles: {paper_estimate.lookup_cycles:,}")
print(f"  total cycles: {paper_estimate.total_cycles:,}")
print(f"  physical qubits: {paper_estimate.physical_qubits:,}")
print(f"  runtime: {paper_estimate.runtime_hours:.2f} h")
print(f"  proxy retry risk: {paper_estimate.retry_risk:.3f}")

#%%
