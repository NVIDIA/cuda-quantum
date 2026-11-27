# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Argument resolution for compatible public terminology aliases."""

UNSET = object()


def resolve_alias(value, legacy, *, name, legacy_name, default=UNSET):
    """Choose one spelling and reject ambiguous calls before construction."""

    if value is not UNSET and legacy is not UNSET:
        raise TypeError(f"pass {name}= or legacy {legacy_name}=, not both")
    result = value if value is not UNSET else legacy
    if result is UNSET:
        if default is UNSET:
            raise TypeError(f"missing required argument: {name!r}")
        return default
    return result
