# CUDA-Q Logical examples

The top-level sequence is the primary path through CUDA-Q Logical. It starts
with the smallest useful estimate, descends through progressively more of the
compilation stack, then applies that workflow to larger kernels and a complete
user-authored target. Each example authors an ordinary `@cudaq.kernel`, selects
a target stack, and calls `cudaq.estimate`; compilation details stay inside the
target.

1. `00_logical_resource_estimate.py` is the minimal introduction: author a
   kernel and read its portable logical resources from `estimate.resources`.
2. `01_clifford_t_resource_estimate.py` descends one layer, changing rotation
   precision and reading the resulting Clifford+T resources.
3. `02_surface_code_resource_estimate.py` descends through surface-code and
   physical compilation. It varies layout parameters, then holds the layout
   fixed to expose operating-point effects.
4. `03_fermi_hubbard.py` moves beyond teaching-scale kernels and estimates a
   Trotterized Fermi--Hubbard workload.
5. `04_carbon_code.py` removes the built-in-target convenience and constructs
   the complete stack from scratch: the Carbon `[[12,2,4]]` code, paired
   CUDA-Q kernels used as gadgets, and the logical, QEC, and physical layers.
6. `05_gidney_ekera.py` brings the pieces together on the exact folded
   RSA-2048, RSA-3072, or RSA-4096 resource kernel and a detailed
   factory/application stack. It calls
   `estimate_using_kernel_profile()` directly for the default fast path;
   `estimate_physically()` is the direct, opt-in paper-scale compilation and
   scheduling path.

## Where do the results live?

`cudaq.estimate` returns an `EstimateResult` with two views:

- `estimate.resources` is a CUDA-Q `Resources` object that reports the
  **logical** program only, whenever a LOGICAL estimate is available: gate
  counts (`count`, `count_controls`, `to_dict`), `num_qubits`, `depth`, and
  `t_depth`. Measurements appear as `mz`, preparations as `reset`, and idle
  steps as `idle`. `depth` is a serial upper bound. This is all most examples
  need (00, 01, 03, and the logical part of 04).
- `estimate.annotations[<TIER>]` holds the full result of each tier that was
  computed (`LOGICAL`, `STATIC`, `ANALYTICAL`, `SCHEDULE`). Use it for the
  physical tiers and for logical details `Resources` cannot express, such as
  idle sites, synthesis demand, and resource requests. The typed views
  (`LogicalEstimate`, `FabricCounts`, ...) rehydrate from it with
  `from_annotations`; examples 00 and 01 show this as a "going deeper" step,
  and 05 reads annotations because its analytical model needs those details.

The `standalone/` examples use `cudaq.logical` directly and expose each
compiler step. They do not define CUDA-Q kernels or call `cudaq.estimate`.

The `mlir/` examples use `qlx-opt` and `qlx-translate` directly.

## Gidney--Ekerå physical compilation

The example intentionally offers two methods with different sources of
authority:

| Method | CUDA-Q compilation | Physical model and reported metrics |
| --- | --- | --- |
| `estimate_using_kernel_profile()` | Stops at portable logical resources. | Applies the Gidney--Ekerå paper's closed-form timing and layout equations to compiler-counted logical qubits and Toffoli gates. It is fast, but the physical result is specific to those explicit formulas rather than a compiled schedule. |
| `estimate_physically()` | Compiles the kernel through the target's logical, QEC, and physical layers and creates a native schedule. | Reports physical qubits, event count, and makespan directly from the application schedule annotations. It validates the complete target stack, but the paper-scale compilation can require minutes and substantial memory. |

The first method is not `cudaq.logical.estimate(...,
tier=Tier.ANALYTICAL)`. It deliberately combines `cudaq.estimate` logical
counts with a small, inspectable paper-specific calculation. The second method
lets the CUDA-Q Logical target own the lower tiers and uses only the metrics
returned by `cudaq.estimate`.

The kernel-profile call is enabled by default. Pass `--physical` when running
the file from the command line to select the paper-scale compilation, and
`--bits` to select the modulus size:

```bash
python 05_gidney_ekera.py

python 05_gidney_ekera.py --physical

python 05_gidney_ekera.py --bits 4096 --physical
```

Each size is one row of Table 3 in arXiv:1905.09749. The factory module derives
the kernel and board sizes from that row.

In a notebook, call `estimate_using_kernel_profile(bits)` or
`estimate_physically(bits)` directly. While building the device, the companion
factory module schedules one workload-independent detailed AutoCCZ lane,
estimates its resources, and derives the compact factory model from that same
schedule. The factory and application are separate resource studies: factory
characterization constructs a component of the device, while the application
schedule is the authority for the final RSA metrics. Neither method performs
factor-producing execution.

Physical error rate, surface-code scaling, and cycle timing belong to each
device's operating point. Physical estimation reads them from the selected
device by default; callers can still override them explicitly for sensitivity
studies. The failure budget remains an estimate policy because it describes
the question being asked of that device, so the target passes it separately.
