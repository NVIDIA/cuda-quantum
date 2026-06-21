# MKL-Q Release Policy

MKL-Q is currently a source-only public CUDA-Q-compatible fork. This policy
defines what is allowed now, what is forbidden now, and what must be true before
future tags, GitHub Releases, wheels, PyPI packages, installers, or signed
artifacts are considered.

## Current Policy

The current public branch is source-only:

- No public version tags.
- No GitHub Releases.
- No PyPI packages.
- No wheels.
- No binary installers.
- No Homebrew formula.
- No signed release artifacts.
- No release tarballs beyond GitHub's automatic source archive for the branch.

Use commit SHAs and the tracked source tree as the public reference. Do not
describe a commit as an MKL-Q release unless this policy has been updated and
the release gates below have passed.

## Allowed Now

These actions are allowed during the source-only phase:

- Publish source commits to `main`.
- Keep `wuls968/MKL-Q` as a fork of `NVIDIA/cuda-quantum`.
- Publish source documentation under `docs/mklq/`.
- Publish sanitized benchmark summaries under `benchmarks/mklq/reports/`.
- Run the lightweight public hygiene workflow.
- Use local ignored JSON under `benchmarks/mklq/results/` for development
  evidence.

## Forbidden Now

Do not do these without a reviewed release plan:

- Create or push release tags.
- Create GitHub Releases.
- Upload wheels, installers, archives, checksums, or signing artifacts.
- Publish to PyPI.
- Publish a Homebrew formula or other package-manager recipe.
- Claim binary compatibility or cross-machine performance certification.
- Make `mklq-metal` the default target.
- Describe `mklq-metal` as full Metal-native or release-ready.

## Release Branch Or Tag Criteria

Before creating a release branch or tag, all of these must be true:

- `docs/mklq/public-release-checklist.md` is complete.
- `docs/mklq/upstream-sync.md` has been followed if upstream was synced.
- `docs/mklq/testing-matrix.md` covers the release gate and any new target
  behavior.
- `python3 benchmarks/mklq/run_correctness_gate.py --install-prefix
  "${HOME}/.cudaq-mklq" --build-dir build-python` passes on Apple Silicon.
- `cmake --build build-python --target install -j 6` passes for the intended
  install prefix.
- GitHub `MKL-Q public hygiene` passes for the exact commit.
- `git status --short --branch` is clean before collecting release evidence.
- No raw benchmark JSON, build output, caches, `.DS_Store`, local signing
  objects, private paths, tokens, or secrets are tracked.
- Any performance claim is backed by sanitized benchmark summaries and states
  machine scope.
- `mklq-metal` support language remains experimental unless a separate Metal
  release-readiness plan has passed.

If tags are introduced later, use an MKL-Q-specific tag namespace such as
`mklq-vX.Y.Z` so tags cannot be confused with upstream CUDA-Q tags.

## GitHub Release Entry Criteria

A GitHub Release may only be considered after the release branch or tag criteria
pass and a release plan also defines:

- release notes;
- supported platforms;
- exact source commit and tag;
- validation commands and results;
- known limitations;
- artifact list, if any;
- checksum and signing policy, if artifacts exist;
- rollback or yanking procedure;
- license and NOTICE review.

Do not attach local build products or ad-hoc artifacts to GitHub Releases.

## Wheel And PyPI Entry Criteria

Wheels or PyPI packages require a separate packaging plan. At minimum, that plan
must define:

- package name and ownership;
- whether the public Python namespace remains `cudaq`;
- Python version support;
- macOS ARM64 build environment;
- reproducible build commands;
- isolated install smoke tests in a fresh virtual environment;
- `nvq++` packaging behavior;
- dynamic library layout and signing behavior;
- dependency and license audit;
- uninstall and upgrade behavior;
- security contact and yanking process.

Do not publish a package that could be mistaken for an official NVIDIA CUDA-Q
package. Do not publish MKL-Q packaging artifacts until this policy is updated
with the accepted plan.

## Binary Artifact Hygiene

The public tree must not track release artifacts during the source-only phase,
including:

- `dist/`
- `wheelhouse/`
- `*.whl`
- `*.dmg`
- `*.pkg`
- `*.zip`
- `*.tar.gz`
- signed local objects;
- local notarization or signing logs.

The public hygiene workflow should reject these if they become tracked.

## Stop Conditions

Stop release or publication work if any of these are true:

- The repository is dirty and the change was not intentionally reviewed.
- `origin/main` does not match the intended release commit.
- GitHub Actions is failing or still unknown for the intended commit.
- The release would publish raw local benchmark payloads.
- The release would publish generated build products or local signing artifacts.
- Documentation claims support that the testing matrix does not prove.
- `mklq-metal` is presented as default-ready without a passed Metal
  release-readiness plan.
- The package or release could be confused with official upstream CUDA-Q.

## Future Policy Updates

When MKL-Q is ready for real release artifacts, update this file in the same
change as the release plan. That update should be reviewed before any tag,
GitHub Release, wheel, PyPI package, installer, or signed artifact is created.
