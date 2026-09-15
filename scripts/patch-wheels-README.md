# Patching cudaq wheels to a new version (e.g. 0.16.0.post1)

Renames the version on a locally-built set of `cudaq` release artifacts
(the `cudaq` metapackage sdist + `cuda-quantum-cu12`/`cu13` wheels) without
rebuilding any code, so PyPI accepts a distinguishable re-upload of the same
content. This does **not** touch `cudaq-logical`, and does **not** upload
anything anywhere -- it only writes patched files to a local output
directory.

## Version format

PyPI enforces [PEP 440](https://peps.python.org/pep-0440/). Use the
post-release segment: `0.16.0.post1`, `0.16.0.post2`, etc. Do **not** use
something like `0.16.0.patch1` -- that's not a valid PEP 440 segment and
PyPI will reject the upload.

## Prerequisites

- `python3` with the `venv` module (standard on any modern Python install).
- The source artifacts in a local directory (see layout below).

## Input layout

Place these files in one directory (default `~/wheels/`) -- this is exactly
what you get by downloading the relevant GitHub Actions artifacts from a
`Deployments`/`Validation` run, unmodified, zipped:

```text
wheels/
  cudaq-metapackage-<VERSION>.zip          # contains cudaq-<VERSION>.tar.gz
  aarch64-cu12-py3.11-wheels.zip           # one .whl each
  aarch64-cu12-py3.12-wheels.zip
  aarch64-cu12-py3.13-wheels.zip
  aarch64-cu12-py3.14-wheels.zip
  aarch64-cu13-py3.11-wheels.zip
  aarch64-cu13-py3.12-wheels.zip
  aarch64-cu13-py3.13-wheels.zip
  aarch64-cu13-py3.14-wheels.zip
  x86_64-cu12-py3.11-wheels.zip
  x86_64-cu12-py3.12-wheels.zip
  x86_64-cu12-py3.13-wheels.zip
  x86_64-cu12-py3.14-wheels.zip
  x86_64-cu13-py3.11-wheels.zip
  x86_64-cu13-py3.12-wheels.zip
  x86_64-cu13-py3.13-wheels.zip
  x86_64-cu13-py3.14-wheels.zip
```

## Usage

1. Open `patch-wheels.sh` and set `ORIG_VER` / `NEW_VER` at the top.
2. Run it:

   ```bash
   bash patch-wheels.sh
   ```

3. Output lands in `~/wheels_patched/`, organized by architecture:

   ```text
   wheels_patched/
     aarch64/   *.whl  (cu12 + cu13, all 4 Python versions)
     x86_64/    *.whl  (cu12 + cu13, all 4 Python versions)
     sdist/     cudaq-<NEW_VER>.tar.gz
   ```

The script creates its own throwaway venv under `/tmp` with `build` and
`wheel` installed -- it doesn't touch your system Python.

## What it actually does

- **sdist**: unzips the metapackage bundle, extracts `cudaq-<VERSION>.tar.gz`,
  overwrites `_version.txt` with the new version, **deletes the existing
  `PKG-INFO`** (important -- see gotcha below), and rebuilds the sdist with
  `CUDAQ_META_SDIST_BUILD=1 python3 -m build . --sdist`.
- **wheels**: for each `cuda-quantum-cuXX` wheel, unzips the artifact,
  `python3 -m wheel unpack`s it, renames the package dir and `.dist-info`
  dir to the new version, edits the `Version:` line in `METADATA`, and
  `python3 -m wheel pack`s it back up. `wheel pack` regenerates `RECORD`
  (file hashes) automatically -- don't hand-edit it.

## Gotcha: stale `PKG-INFO` silently keeps the old version

If you rebuild an sdist from an already-built tarball without deleting its
existing `PKG-INFO` first, `python -m build` uses the old `PKG-INFO` as
authoritative and **silently ignores** the updated `_version.txt` -- you'll
get `cudaq-<VERSION>.tar.gz` again instead of `cudaq-<NEW_VER>.tar.gz`, with
no error. Always delete `PKG-INFO` before rebuilding from an extracted
sdist. (The script already does this.)

## Verifying the output before uploading anywhere

```bash
# Every wheel's METADATA should report the new version
for f in ~/wheels_patched/*/*.whl; do
  unzip -p "$f" "*.dist-info/METADATA" | grep "^Version:"
done

# Real install test (only works for wheels matching your machine's arch)
python3 -m pip install --no-deps --target /tmp/scratch ~/wheels_patched/x86_64/cuda_quantum_cu12-<NEW_VER>-cp312-*.whl

# sdist install test (any arch, exercises the dynamic dependency metadata)
python3 -m venv /tmp/sdist-test && /tmp/sdist-test/bin/pip install --no-deps ~/wheels_patched/sdist/cudaq-<NEW_VER>.tar.gz
/tmp/sdist-test/bin/pip show cudaq
```

## Uploading (not done by this script)

This script never contacts PyPI. When you're ready:

```bash
python3 -m twine upload --repository testpypi ~/wheels_patched/sdist/* ~/wheels_patched/aarch64/* ~/wheels_patched/x86_64/*
# once verified on TestPyPI:
python3 -m twine upload ~/wheels_patched/sdist/* ~/wheels_patched/aarch64/* ~/wheels_patched/x86_64/*
```
