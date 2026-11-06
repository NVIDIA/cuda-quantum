# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Save the builds of one estimate to files and load them back.

``cudaq.estimate(kernel, save_temps=directory)`` writes every build a logical
target produces on the way to its estimates. Each build is written twice:

* ``<kernel>.<stage>.<sha>.mlir`` is its MLIR, for reading, and
* ``<kernel>.<stage>.<sha>.build`` is its replay bundle, which
  :func:`load_build` reads back into an identical :class:`Build`.

``<sha>`` is the first 12 hexadecimal digits of :attr:`Build.content_sha256`.
The MLIR alone cannot be loaded: a placed or encoded build also commits to the
placement and QEC selection witnesses that its replay bundle carries.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .build import Build

# The annotation of an `EstimateResult` that locates the saved builds.
BUILDS_ANNOTATION = "BUILDS"

# The stages a target can save a build for, in compilation order. The
# annotation does not preserve the order in which they were saved.
_STAGE_ORDER = ("p0", "clifford_t", "p1", "p2", "p3")


class BuildSaver:
    """Writes the builds of one estimate into a directory."""

    def __init__(self, directory) -> None:
        self.directory = Path(os.fspath(directory))
        self._saved: dict[str, dict[str, str]] = {}
        self._digests: set[str] = set()
        self._kernel: str | None = None

    def save(self, stage: str, build: Build) -> None:
        """Save ``build`` as the product of ``stage``, unless already saved."""

        digest = build.content_sha256
        if digest in self._digests:
            return
        self._digests.add(digest)
        # Later stages rename their root, so name every file after the kernel.
        if self._kernel is None:
            self._kernel = build.root.symbol
        stem = (f"{self._kernel}.{stage}."
                f"{digest.removeprefix('sha256:')[:12]}")
        self.directory.mkdir(parents=True, exist_ok=True)
        mlir, bundle = (self.directory / f"{stem}.mlir",
                        self.directory / f"{stem}.build")
        mlir.write_text(build.to_mlir())
        build.serialize(bundle)
        self._saved[stage] = {"mlir": str(mlir), "build": str(bundle)}

    def manifest(self) -> dict[str, dict[str, str]]:
        """The saved files by stage, as reported in the estimate annotations."""

        return dict(self._saved)


def load_build(source, stage: str | None = None) -> Build:
    """Load a build saved by ``cudaq.estimate(..., save_temps=directory)``.

    Args:
      source: The ``EstimateResult`` of the estimate, or the path of one of its
        ``.build`` files.
      stage: Which of the estimate's builds to load, such as ``"p0"`` or
        ``"p3"``. Defaults to the last stage the target compiled to. Only valid
        with an ``EstimateResult``.

    Returns:
      The :class:`Build`, identical to the one the target produced: it has the
      same :attr:`Build.content_sha256`.
    """

    if isinstance(source, (str, os.PathLike)):
        if stage is not None:
            raise TypeError("stage= only applies to an EstimateResult; a path "
                            "names one build")
        return Build.replay(Path(os.fspath(source)))

    annotations = getattr(source, "annotations", None)
    if not isinstance(annotations, Mapping):
        raise TypeError(
            "load_build expects an EstimateResult or the path of a .build file, "
            f"not {type(source).__name__}")
    saved = annotations.get(BUILDS_ANNOTATION)
    if not saved:
        raise ValueError(
            "the estimate saved no builds; pass save_temps=<directory> to "
            "cudaq.estimate")
    if stage is None:
        stage = max(saved, key=_stage_rank)
    if stage not in saved:
        raise ValueError(f"the estimate saved no {stage!r} build; it saved: " +
                         ", ".join(sorted(saved, key=_stage_rank)))
    return Build.replay(Path(saved[stage]["build"]))


def _stage_rank(stage: str) -> int:
    try:
        return _STAGE_ORDER.index(stage)
    except ValueError:
        return len(_STAGE_ORDER)
