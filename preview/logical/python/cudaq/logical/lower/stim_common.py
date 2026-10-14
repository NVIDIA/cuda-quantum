# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations
import ast

from .._contracts.projection import (
    CompiledInterfaceManifest,
    ProjectedMeasurement,
    ProjectedPort,
)


def _text(attribute):
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


def _symbol(operation):
    try:
        return _text(operation.attributes["sym_name"])
    except KeyError:
        return None


def _partition(attribute):
    text = str(attribute)
    return text[text.find("<") + 1:text.rfind(">")]


def _pairs(attribute, control_width, target_width):
    """Parse Fabric's canonical bounded matching."""

    text = _text(attribute)
    if text == "index":
        return tuple(
            (index, index) for index in range(min(control_width, target_width)))
    if ":" in text:
        result = []
        for entry in text.split(","):
            control, target = entry.split(":", 1)
            result.append((int(control.strip()), int(target.strip())))
        return tuple(result)
    return tuple(ast.literal_eval(text))


def _code_name(type_):
    text = str(type_)
    prefix = "!fabric.patch<@"
    if not text.startswith(prefix):
        return None
    return text[len(prefix):-1].split(",", 1)[0].strip().lstrip("@")


def _is_patch_like(type_):
    text = str(type_)
    return text.startswith("!fabric.patch<@") or text.startswith(
        "!fabric.patch_frame<@")
