# MKL-Q Examples

These examples are small source-only smoke programs for the public MKL-Q
targets. Build and install MKL-Q first, then run them from the repository root.

## Python

```bash
PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/bell.py --target mklq-cpu --target mklq-metal

PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/ghz.py --target mklq-cpu --target mklq-metal

PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/parametric.py --target mklq-cpu --target mklq-metal

PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/phase_kickback.py --target mklq-cpu --target mklq-metal

PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/clifford_chain.py --target mklq-cpu --target mklq-metal
```

Use `--shots N` to change the sample count.

## One-command Verification

Run all Python and C++ examples for both public MKL-Q targets:

```bash
PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/verify_examples.py \
  --install-prefix "${HOME}/.cudaq-mklq"
```

The verifier writes ignored JSON under `benchmarks/mklq/results/` and checks
that each example only reports its expected bitstring support. The current
fixture set covers Bell/GHZ entanglement, parameterized rotations,
controlled-phase kickback, and a deterministic Clifford chain.

## C++

```bash
"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-cpu \
  examples/mklq/cpp/bell.cpp -o /tmp/mklq_bell_cpu
/tmp/mklq_bell_cpu 100

"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-metal \
  examples/mklq/cpp/bell.cpp -o /tmp/mklq_bell_metal
/tmp/mklq_bell_metal 100

"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-cpu \
  examples/mklq/cpp/ghz.cpp -o /tmp/mklq_ghz_cpu
/tmp/mklq_ghz_cpu 100

"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-metal \
  examples/mklq/cpp/ghz.cpp -o /tmp/mklq_ghz_metal
/tmp/mklq_ghz_metal 100

"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-cpu \
  examples/mklq/cpp/parametric.cpp -o /tmp/mklq_parametric_cpu
/tmp/mklq_parametric_cpu 100

"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-cpu \
  examples/mklq/cpp/phase_kickback.cpp -o /tmp/mklq_phase_kickback_cpu
/tmp/mklq_phase_kickback_cpu 100

"${HOME}/.cudaq-mklq/bin/nvq++" --target mklq-cpu \
  examples/mklq/cpp/clifford_chain.cpp -o /tmp/mklq_clifford_chain_cpu
/tmp/mklq_clifford_chain_cpu 100
```

The Metal target is experimental. If a backend operation is outside the current
resident Metal path, MKL-Q may fall back to the CPU oracle as documented in
`docs/mklq/known-limitations.md`.
