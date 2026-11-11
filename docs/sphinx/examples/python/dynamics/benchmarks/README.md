## Overview

CUDA-Q `evolve` benchmarks on the `dynamics` target:

- `heisenberg_model.py`: Heisenberg spin chain of length `--spins` with
  dissipation.
- `cavity_qed.py`: Jaynes-Cummings model with a cavity Fock space of dimension
  `--fock_dim`.

Each script runs the evolution four times with the same step size,
`--t_final / --n_steps`: two of 1 step, one of 3 steps, and the measured
evolution of `--n_steps` steps. Before each evolution the ranks synchronize,
and each reported time is that of the slowest rank. The scripts print:

- `Time per step`: time per evolution step without startup/initialization
- `Evolve initialization time`: the fixed setup and tear-down cost that every
  evolution pays.
- `Startup time`: The one-time cost of the first evolution in a process.

Run either script with `--help` to list the model and timing options.

## Usage

```bash
python3 heisenberg_model.py
python3 cavity_qed.py
```

The defaults (`--spins 8` or `--fock_dim 64`) are intended for quick
validation of an installation; each script finishes in under 20 seconds on a
single GPU, mostly in one-time initialization. They are too small to measure
performance.

The scripts use MPI only when run with `python3 -m mpi4py`, which requires
`mpi4py` to be installed; otherwise they run on a single GPU without MPI. The
first line of output says which. To distribute the state across multiple
GPUs, launch with MPI:

```bash
mpiexec -np <ranks> python3 -m mpi4py heisenberg_model.py --spins 16
```

## Sizing for benchmarking

Both models evolve a density matrix of dimension `D`, where `D = 2^spins` for
the Heisenberg chain and `D = 2 * fock_dim` for the cavity model. One complex
double precision density matrix occupies `16 D^2` bytes:

| Heisenberg `--spins` | Cavity `--fock_dim` | Density matrix |
| -------------------- | ------------------- | -------------- |
| 12                   | 2048                | 256 `MiB`      |
| 14                   | 8192                | 4 `GiB`        |
| 15                   | 16384               | 16 `GiB`       |
| 16                   | 32768               | 64 `GiB`       |
| 17                   | 65536               | 256 `GiB`      |
| 18                   | 131072              | 1 `TiB`        |

The integrator and the distributed operator action need several state-sized
buffers, so budget about eight density matrices plus a few `GiB` of overhead.
For the Heisenberg model that budget is shared across all GPUs: with MPI the
density matrix is distributed, so the largest `--spins` grows with the GPU
count. The cavity model builds its initial density matrix with CuPy on every
rank before it is distributed, so its largest `--fock_dim` is limited by the
memory of a single GPU regardless of the GPU count.

For representative timings:

- Choose the largest size that fits the selected GPUs.
- Keep `--n_steps` large enough that the measured evolution is much longer
  than the 3-step evolution, so that timing noise is small relative to the
  difference between them.

### Example: NVIDIA GB200

These are the largest sizes that fit the budget above on GB200 GPUs with
192 `GiB` each, run for 30 steps. Adding GPUs raises the Heisenberg chain by
one spin for every fourfold increase in GPU count, while the cavity size
stays at the single-GPU limit.

One GPU:

```bash
python3 heisenberg_model.py --spins 15 --n_steps 30
python3 cavity_qed.py --fock_dim 16384 --n_steps 30
```

Four GPUs:

```bash
mpiexec -np 4 python3 -m mpi4py heisenberg_model.py --spins 16 --n_steps 30
mpiexec -np 4 python3 -m mpi4py cavity_qed.py --fock_dim 16384 --n_steps 30
```
