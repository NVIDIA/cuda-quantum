# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Private, domain-independent implementation foundations."""

from .immutable import ImmutableValue, freeze_mapping, freeze_value

__all__ = ["ImmutableValue", "freeze_mapping", "freeze_value"]
