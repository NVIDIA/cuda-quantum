# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Ordered syndrome-extraction layers from ``code.colored_schedule``."""

from __future__ import annotations

import cudaq.logical


def test_colored_schedule_is_plain_data_covering_every_incidence_once():
    code = cudaq.logical.codes.rotated_surface(3)
    x_layers, z_layers = code.colored_schedule(x_order=(0, 1, 2, 3))
    # plain nested tuples, not a bespoke type
    assert isinstance(x_layers, tuple) and isinstance(z_layers, tuple)
    # every (check, data) incidence appears exactly once
    x_flat = sorted(pair for layer in x_layers for pair in layer)
    expected = sorted((check, data)
                      for check, support in enumerate(code.hx)
                      for data in support)
    assert x_flat == expected
    # each layer is a matching (no ancilla or data qubit twice)
    for layer in (*x_layers, *z_layers):
        checks = [c for c, _ in layer]
        data = [d for _, d in layer]
        assert len(set(checks)) == len(checks) and len(set(data)) == len(data)
