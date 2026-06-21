# MKL-Q Examples

These examples are small source-only smoke programs for the public MKL-Q
targets. Build and install MKL-Q first, then run them from the repository root.

## Python

```bash
PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/bell.py --target mklq-cpu --target mklq-metal

PYTHONPATH="${HOME}/.cudaq-mklq" \
python3 examples/mklq/python/ghz.py --target mklq-cpu --target mklq-metal
```

Use `--shots N` to change the sample count.

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
```

The Metal target is experimental. If a backend operation is outside the current
resident Metal path, MKL-Q may fall back to the CPU oracle as documented in
`docs/mklq/known-limitations.md`.
