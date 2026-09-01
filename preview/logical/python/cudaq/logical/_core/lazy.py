# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Small helper for initialization-safe public domain facades."""

from __future__ import annotations

from importlib import import_module
from typing import Mapping


def resolve(namespace: dict, exports: Mapping[str, str], name: str):
    """Resolve and memoize one ``"module:attribute"`` facade export."""

    target = exports.get(name)
    if target is None:
        raise AttributeError(name)
    package = namespace["__package__"]
    if ":" in target:
        module_name, attribute = target.split(":", 1)
        value = getattr(import_module(module_name, package), attribute)
    else:
        value = import_module(target, package)
    namespace[name] = value
    return value


def public_dir(namespace: Mapping[str, object], exports: Mapping[str, str]):
    return sorted(set(namespace) | set(exports))
