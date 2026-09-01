# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared immutable-artifact support for compiler mapping seams.

Stage-specific mapping APIs own their semantic problem and plan types.  This
module only supplies the representation-independent mechanics they share:
canonical JSON, recursive immutability, content digests, persistence, and the
problem/plan provenance check.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class MappingVerificationError(ValueError):
    """A mapping artifact is stale, malformed, or belongs to another problem."""


def freeze_json(value: Any) -> Any:
    """Return a recursively immutable JSON value.

    Mapping keys are normalized to strings and sorted.  Sorting here is not
    required by ``MappingProxyType``; it makes iteration deterministic too.
    """

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("mapping artifact keys must be strings")
        return MappingProxyType(
            {key: freeze_json(value[key]) for key in sorted(value)})
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("mapping artifacts contain only JSON values, got "
                    f"{type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return an ordinary JSON-serializable copy of a frozen artifact value."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize one artifact value with a stable byte-level representation."""

    return json.dumps(
        thaw_json(freeze_json(value)),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def artifact_digest(value: Any) -> str:
    """Return the canonical SHA-256 digest for one artifact payload."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def verify_artifact_digest(
    payload: Mapping[str, Any],
    claimed_digest: str,
    *,
    artifact: str,
) -> None:
    """Fail closed when a serialized artifact's content digest is stale."""

    actual = artifact_digest(payload)
    if claimed_digest != actual:
        raise MappingVerificationError(
            f"{artifact} digest mismatch: expected {claimed_digest!r}, "
            f"computed {actual!r}")


def verify_plan_for_problem(*, problem_digest: str,
                            plan_problem_digest: str) -> None:
    """Require a plan to identify the exact problem it solves."""

    if plan_problem_digest != problem_digest:
        raise MappingVerificationError(
            "mapping plan belongs to a different problem: "
            f"plan has {plan_problem_digest!r}, problem has {problem_digest!r}")


def save_artifact(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist canonical JSON with one trailing newline."""

    Path(path).write_text(canonical_json(payload) + "\n", encoding="utf-8")


def load_artifact(path: str | Path) -> Mapping[str, Any]:
    """Load one JSON object without assigning it semantic meaning."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MappingVerificationError(
            "mapping artifact root must be a JSON object")
    return value


__all__ = [
    "MappingVerificationError",
    "artifact_digest",
    "canonical_json",
    "freeze_json",
    "load_artifact",
    "save_artifact",
    "thaw_json",
    "verify_artifact_digest",
    "verify_plan_for_problem",
]
