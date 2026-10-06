# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonical codecs and commitment helpers for lattice-surgery artifacts."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping

from cudaq.logical.stages import (
    P2,
    P3,
)


def _mapping_module():
    from ...compiler import mapping

    return mapping


def _json_value(value):
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(
                "lattice-surgery artifact mapping keys must be strings")
        return {
            key: _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            "lattice-surgery artifacts require finite floating-point values")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "lattice-surgery artifacts contain only canonical JSON values, got "
        f"{type(value).__name__}")


def _frozen_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _mapping_module().freeze_json(_json_value(dict(values or {})))


def _digest(value) -> str:
    return _mapping_module().artifact_digest(_json_value(value))


def _nonempty_pipeline_name(value, *, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a nonempty string")
    if not value:
        raise ValueError(f"{what} must be a nonempty string")
    return value


def _pipeline_options(options) -> list[list[Any]]:
    records = []
    keys = set()
    for option in options:
        if (isinstance(option, (str, bytes)) or
                not isinstance(option, (tuple, list)) or len(option) != 2):
            raise TypeError(
                "lattice-surgery PassSpec options must be key/value pairs")
        key, value = option
        key = _nonempty_pipeline_name(
            key, what="lattice-surgery PassSpec option key")
        if key in keys:
            raise ValueError(
                f"duplicate lattice-surgery PassSpec option key {key!r}")
        keys.add(key)
        records.append([key, _json_value(value)])
    return records


def _pipeline_facets(values, *, field: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"lattice-surgery PassSpec {field} must be a sequence of names")
    facets = []
    seen = set()
    for value in values:
        facet = _nonempty_pipeline_name(
            value, what=f"lattice-surgery PassSpec {field} facet name")
        if facet in seen:
            raise ValueError(
                f"duplicate lattice-surgery PassSpec {field} facet "
                f"name {facet!r}")
        seen.add(facet)
        facets.append(facet)
    return facets


def _pipeline_record(pipeline) -> dict[str, Any]:
    """Return the canonical identity of one P2/P3 compiler recipe."""

    from ...compiler import PassSpec, Pipeline

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "lattice-surgery compilers must contribute a typed cudaq.logical.Pipeline"
        )
    if pipeline.output_stage not in {P2, P3}:
        raise TypeError("lattice-surgery recipes must produce P2 or P3")
    passes = []
    for item in pipeline.passes:
        if not isinstance(item, PassSpec):
            raise TypeError(
                "lattice-surgery pipelines require typed cudaq.logical.PassSpec values"
            )
        passes.append({
            "name":
                _nonempty_pipeline_name(
                    item.name,
                    what="lattice-surgery PassSpec name",
                ),
            "options":
                _pipeline_options(item.options),
            "requires_facets":
                _pipeline_facets(
                    item.requires_facets,
                    field="requires_facets",
                ),
            "provides_facets":
                _pipeline_facets(
                    item.provides_facets,
                    field="provides_facets",
                ),
            "preserves_facets":
                _pipeline_facets(
                    item.preserves_facets,
                    field="preserves_facets",
                ),
            "invalidates_facets":
                _pipeline_facets(
                    item.invalidates_facets,
                    field="invalidates_facets",
                ),
            "recomputes_facets":
                _pipeline_facets(
                    item.recomputes_facets,
                    field="recomputes_facets",
                ),
        })
    return {
        "output_profile": str(pipeline.output_profile),
        "passes": passes,
    }


def _pipeline_digest(pipeline) -> str:
    return _digest(_pipeline_record(pipeline))


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is not supported")


def _strict_json_loads(text: str):
    return json.loads(text, parse_constant=_reject_json_constant)


def _require_digest(value, *, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{what} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{what} must be a hexadecimal SHA-256 digest") from exc
    return value


def _require_commitment_digest(value, *, what: str) -> str:
    """Accept canonical bare or ``sha256:``-qualified commitments."""

    payload = (value[len("sha256:"):] if isinstance(value, str) and
               value.startswith("sha256:") else value)
    _require_digest(payload, what=what)
    return value


def _record(
    value,
    *,
    what: str,
    keys: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{what} must be a JSON object")
    expected = set(keys)
    if set(value) != expected:
        raise ValueError(f"{what} fields must be exactly {sorted(expected)!r}")
    return value


def _array(value, *, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{what} must be a JSON array")
    return value


__all__ = [
    "_array",
    "_digest",
    "_frozen_mapping",
    "_json_value",
    "_mapping_module",
    "_pipeline_digest",
    "_pipeline_facets",
    "_pipeline_options",
    "_pipeline_record",
    "_record",
    "_require_commitment_digest",
    "_require_digest",
    "_strict_json_loads",
]
