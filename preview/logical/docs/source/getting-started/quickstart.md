# Quick start

CUDA-Q Logical expands CUDA-Q with an open, extensible logical layer for
developing and evaluating fault-tolerant quantum applications. It lets you keep
a workload fixed while comparing QEC codes and system architectures. This quick
start walks through installation and a first logical resource estimate: define
the same Bell-pair workload as a CUDA-Q kernel or a standalone logical program,
and obtain identical resource counts.

## Use a CUDA-Q kernel

Install [CUDA-Q][cudaq-install], which includes CUDA-Q Logical:

```bash
pip install cudaq
```

Once installed, you can write a quantum program as a CUDA-Q kernel and use
CUDA-Q Logical to estimate its resources at different compilation layers. Start
with logical-qubit and operation counts; with a QEC realization and device
configured, you can also estimate physical qubits and runtime. Use CUDA-Q
kernels when you want to build on CUDA-Q's libraries and infrastructure or
estimate an existing CUDA-Q application.

The [kernel example][kernel-example] below defines a kernel that allocates two
qubits, applies a Hadamard and a controlled-X to prepare a Bell pair, and
measures both qubits in the Z basis. It selects `cql.targets.estimator`, which
stops at a portable logical representation of the program, then calls
`cudaq.estimate(bell_pair)` to report the peak logical-qubit count and the
Hadamard and controlled-X action counts:

```{literalinclude} ../../../examples/00_logical_resource_estimate.py
:language: python
:start-at: import cudaq
:caption: examples/00_logical_resource_estimate.py
```

To run this example, copy the code into `bell_kernel.py` and run
`python3 bell_kernel.py`. The example is also available in a checked-out
repository; from its root, run
`python3 preview/logical/examples/00_logical_resource_estimate.py`.

The program prints:

```text
Logical Bell-pair resources:
  peak logical qubits: 2
  Hadamard actions: 1
  controlled-X actions: 1
```

## Write a standalone logical program

Use the standalone `cudaq.logical` API when integrating other frontend tools
without depending on the CUDA-Q kernel frontend. Both authoring routes support
placement, QEC codes, gadgets, and the same compilation and estimation stages.

To install CUDA-Q Logical separately, run:

```bash
pip install "cudaq-logical[cu13]"
```

Use `[cu12]` for CUDA 12. These extras include the CUDA-Q runtime and kernel API,
so both examples work with either extra. The bare `cudaq-logical` package
requires an existing CUDA-Q runtime. If you installed `cudaq` above, no second
install is needed.

The [standalone example][standalone-example] defines the same Bell-pair
preparation and measurement using logical operations. It explicitly compiles the
program with `cql.compile(bell)`, then asks `cql.estimate` for a logical-tier
estimate that reports the same resource counts as the kernel example:

```{literalinclude} ../../../examples/standalone/00_logical_program.py
:language: python
:start-at: import cudaq.logical as cql
:caption: examples/standalone/00_logical_program.py
```

To run this example, copy the code into `bell_logical.py` and run
`python3 bell_logical.py`. The example is also available in a checked-out
repository; from its root, run
`python3 preview/logical/examples/standalone/00_logical_program.py`.

The program prints:

```text
Logical Bell-pair resources:
  peak logical qubits: 2
  Hadamard actions: 1
  controlled-X actions: 1
```

Both examples estimate the logical resources needed to prepare and measure a
Bell pair. The output reports resource counts, not sampled measurement results.

## Build from source

Use the nightly CUDA-Q development container; see
[CUDA-Q development setup][development-setup] for setup instructions. From the
repository root, build and install CUDA-Q and CUDA-Q Logical with:

```bash
bash scripts/build_cudaq.sh -L
```

For prerequisites and other source-installation options, see
[CUDA-Q installation from source][source-install].

## Where to go next

These are logical counts; neither example chooses a QEC code or a device. For
placement, QEC, and physical estimates, continue with
[The logical programming stack](cudaq-logical-in-practice.md) or browse the
[examples](../use-cases/examples/index.md).

[cudaq-install]:
https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q

[development-setup]:
https://github.com/NVIDIA/cuda-quantum/blob/main/Dev_Setup.md#working-inside-the-provided-development-container-recommended

[source-install]:
https://nvidia.github.io/cuda-quantum/latest/using/install/data_center_install.html

[kernel-example]:
https://github.com/NVIDIA/cuda-quantum/blob/main/preview/logical/examples/00_logical_resource_estimate.py

[standalone-example]:
https://github.com/NVIDIA/cuda-quantum/blob/main/preview/logical/examples/standalone/00_logical_program.py
