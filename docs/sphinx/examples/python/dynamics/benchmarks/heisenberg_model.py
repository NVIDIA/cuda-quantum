# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Heisenberg model master equation solver benchmark
# The benchmark models follow those in qutip-benchmark
# (https://github.com/qutip/qutip-benchmark).

import math
import sys
import numpy as np
import cupy as cp
import time
import argparse

# Parameters
parser = argparse.ArgumentParser(
    description="CUDA-Q dynamics benchmark with Heisenberg model")
parser.add_argument("--spins",
                    type=int,
                    default=8,
                    help="Length of the spin chain")
parser.add_argument("--gamma", type=float, default=1.0, help="Decay rate")
parser.add_argument("--J_x",
                    type=float,
                    default=2.5,
                    help="XX interaction strength")
parser.add_argument("--J_y",
                    type=float,
                    default=1.75,
                    help="YY interaction strength")
parser.add_argument("--J_z",
                    type=float,
                    default=3.25,
                    help="ZZ interaction strength")
parser.add_argument("--h",
                    type=float,
                    default=1.0,
                    help="Self-energy (Z) coefficient")
parser.add_argument("--t_final",
                    type=float,
                    default=1.0,
                    help="Final time of the measured evolution")
parser.add_argument("--n_steps",
                    type=int,
                    default=1000,
                    help="Number of time steps in the measured evolution")

params = parser.parse_args()
if params.n_steps <= 3:
    parser.error("--n_steps must be greater than 3")

spins = params.spins
t_final = params.t_final
n_steps = params.n_steps
step_size = t_final / n_steps

# Heisenberg model params
Jx = params.J_x
Jy = params.J_y
Jz = params.J_z
h = params.h
gamma = params.gamma

import cudaq
from cudaq import spin, Schedule, RungeKuttaIntegrator
# Use MPI only when the script is run with `python3 -m mpi4py`
use_mpi = "mpi4py" in sys.modules
if use_mpi:
    from mpi4py import MPI
    cudaq.mpi.initialize()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()
else:
    rank = 0
    num_ranks = 1


# Print only from rank 0.
def log(*args):
    if rank == 0:
        print(*args, flush=True)


if use_mpi:
    log(f"MPI: enabled, {num_ranks} task(s)")
else:
    log("MPI: not used (run with `python3 -m mpi4py` to enable)")

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")
dimensions = {i: 2 for i in range(spins)}

# Construct the Hamiltonian
H = h * spin.z(0)
for i in range(spins - 1):
    H += h * spin.z(i + 1)

for i in range(spins - 1):
    H += Jx * spin.x(i) * spin.x(i + 1)
    H += Jy * spin.y(i) * spin.y(i + 1)
    H += Jz * spin.z(i) * spin.z(i + 1)

collapse_operators = [np.sqrt(gamma) * spin.x(i) for i in range(spins)]

# Initial state vector
initial_state = cudaq.dynamics.InitialState.ZERO


# Every evolution uses the same step size, so each step does the same work.
# With MPI, returns the slowest rank's elapsed time.
def timed_evolve(steps):
    times = np.linspace(0.0, step_size * steps, steps + 1)
    if use_mpi:
        comm.Barrier()
    start = time.perf_counter()
    cudaq.evolve(H,
                 dimensions,
                 Schedule(times, ["time"]),
                 initial_state,
                 observables=[],
                 collapse_operators=collapse_operators,
                 store_intermediate_results=cudaq.IntermediateResultSave.NONE,
                 integrator=RungeKuttaIntegrator(order=1))
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
    if use_mpi:
        elapsed = comm.allreduce(elapsed, op=MPI.MAX)
    return elapsed


# Format a time in seconds as milliseconds with 3 significant digits.
def format_ms(seconds):
    value = seconds * 1000.0
    if value == 0.0:
        return "0 ms"
    exponent = math.floor(math.log10(abs(value)))
    value = round(value, 2 - exponent)
    exponent = math.floor(math.log10(abs(value)))
    return f"{value:.{max(0, 2 - exponent)}f} ms"


# Run the simulation
log(f"Running benchmark, spins = {spins}, n_steps = {n_steps}")
first_time = timed_evolve(1)
second_time = timed_evolve(1)
baseline_time = timed_evolve(3)
run_time = timed_evolve(n_steps)

time_per_step = (run_time - baseline_time) / (n_steps - 3)
evolve_init_time = baseline_time - 3 * time_per_step
startup_time = (first_time + second_time - 2 *
                (evolve_init_time + time_per_step))
log(f"Time per step: {format_ms(time_per_step)}")
log(f"Evolve initialization time: {format_ms(evolve_init_time)}")
log(f"Startup time: {format_ms(startup_time)}")
if use_mpi:
    cudaq.mpi.finalize()
