# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from cudaq.logical.experiments.definition import Experiment
from .build import Build


@dataclass(frozen=True, slots=True)
class ExperimentBundle:
    """Immutable collection of independently replayable experiment builds."""

    builds: tuple[Build, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "builds", tuple(self.builds))
        if not self.builds:
            raise ValueError("ExperimentBundle requires at least one build")
        if any(build.experiment is None for build in self.builds):
            raise ValueError("every bundled build must contain an experiment")

    @property
    def experiments(self):
        return tuple(build.experiment for build in self.builds)

    def __len__(self):
        return len(self.builds)

    def __iter__(self):
        return iter(self.builds)

    def __getitem__(self, index):
        return self.builds[index]

    def serialize(self, path=None) -> bytes:
        payload = json.dumps(
            {
                "schema":
                    "qlx.experiment-bundle/v1",
                "builds": [
                    json.loads(build.serialize().decode("utf-8"))
                    for build in self.builds
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if path is not None:
            Path(path).write_bytes(payload)
        return payload

    @classmethod
    def replay(cls, payload):
        if isinstance(payload, (str, Path)):
            payload = Path(payload).read_bytes()
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                "ExperimentBundle.replay expects bytes or a filesystem path")
        envelope = json.loads(bytes(payload).decode("utf-8"))
        if envelope.get("schema") != "qlx.experiment-bundle/v1":
            raise ValueError("unsupported experiment-bundle schema")
        return cls(
            tuple(
                Build.replay(
                    json.dumps(build, sort_keys=True, separators=(
                        ",", ":")).encode("utf-8"))
                for build in envelope.get("builds", ())))


def compile_many(experiments, *, pipeline=None, **kwargs) -> ExperimentBundle:
    from .compile import compile

    experiments = tuple(experiments)
    if not experiments:
        raise ValueError(
            "cudaq.logical.compile_many requires at least one experiment")
    if any(not isinstance(value, Experiment) for value in experiments):
        raise TypeError(
            "cudaq.logical.compile_many expects cudaq.logical.Experiment values"
        )
    return ExperimentBundle(
        tuple(
            compile(experiment, pipeline=pipeline, **kwargs)
            for experiment in experiments))


__all__ = ["ExperimentBundle", "compile_many"]
