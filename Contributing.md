# Contributing

Thank you for your interest in MKL-Q. This repository is a CUDA-Q-compatible
Apple Silicon fork, so contributions should preserve CUDA-Q API compatibility
unless a change is explicitly scoped to an MKL-Q target.

## Before Opening A Pull Request

- Open an issue for non-trivial backend, build, or public API changes.
- Keep changes reviewable: separate backend work, benchmark evidence, docs, and
  repository configuration into separate commits or pull requests where
  practical.
- Do not commit build products, raw local benchmark JSON, caches, `.DS_Store`,
  secrets, tokens, or local machine paths.
- Keep `cudaq` as the Python namespace and `nvq++` as the compiler entry point
  for compatibility.

## Local Checks

Run the smallest relevant checks first. For MKL-Q backend changes, these are the
usual gates:

```bash
ctest --test-dir build-python -R "(mklq_(cpu|metal)_MKLQ|backend_target_setter_check|TargetConfigTester)" --output-on-failure

PYTHONPATH="$(pwd)/build-python/python" \
python3 -m pytest \
  python/tests/backends/test_mklq_nvqpp_smoke.py \
  python/tests/backends/test_mklq_benchmark_harness.py \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q
```

Before publishing source changes, also run:

```bash
git diff --check
```

## Commit Sign-off

Contributors should sign off commits to certify that they have the right to
submit the contribution under the Apache License 2.0:

```bash
git commit -s -m "feat: add mklq improvement"
```

This uses the Developer Certificate of Origin:
<https://developercertificate.org/>.

## Upstream Contributions

If a change is generally useful to CUDA-Q rather than specific to MKL-Q, consider
whether it should be proposed upstream to `NVIDIA/cuda-quantum`. MKL-Q keeps
upstream history and compatibility so that such patches remain practical.
