# Standalone CUDA-Q Logical examples

These examples use `cudaq.logical` directly. They do not define
`@cudaq.kernel` functions and do not call `cudaq.estimate`. Instead, they make
the logical authoring, placement, QEC selection, physical lowering, scheduling,
direct estimation, and `cudaq.run` APIs visible.

The sequence progresses from portable logical programs through a complete
physical schedule and Gidney--Ekerå lookup-addition resource estimates.
The execution examples then progress from ideal computation to noise and
measurement-controlled execution:

- `06_steane_bernstein_vazirani.py` recovers a hidden bit string with Steane
  encoded qubits on Stim.
- `07_t_factory_evaluation.py` evaluates T-state distillation with non-Clifford
  gates, measurement feedback, and raw-state noise on `BareQubit` patches.

The encoded example uses an explicit Clifford preparation circuit. It does not
claim fault-tolerant preparation or decoded error correction.
