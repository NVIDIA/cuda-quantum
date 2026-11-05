# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Minimum-depth Tanner-graph edge coloring for CSS schedules."""

import cudaq.logical as cql

_HX = ((0,), (0, 1), (1, 2))
_HZ = ((3,), (3, 4), (4, 5))


def _degree_two_code():
    """Two disjoint five-edge paths with maximum degree two."""

    return cql.codes.CSSCode(
        name="degree_two_coloring",
        n=6,
        k=0,
        hx=_HX,
        hz=_HZ,
    )


def _maximum_degree(checks):
    check_degree = max((len(support) for support in checks), default=0)
    data_degree = {}
    for support in checks:
        for data in support:
            data_degree[data] = data_degree.get(data, 0) + 1
    return max(check_degree, max(data_degree.values(), default=0))


def _assert_optimal_edge_coloring(checks, layers):
    expected = sorted((check, data)
                      for check, support in enumerate(checks)
                      for data in support)
    assert sorted(edge for layer in layers for edge in layer) == expected
    for layer in layers:
        layer_checks = [check for check, _ in layer]
        layer_data = [data for _, data in layer]
        assert len(layer_checks) == len(set(layer_checks))
        assert len(layer_data) == len(set(layer_data))
    assert len(layers) == _maximum_degree(checks)


def test_colored_schedule_is_deterministic_and_minimum_depth_by_default():
    code = _degree_two_code()

    first = code.colored_schedule()
    second = code.colored_schedule()
    explicit_none = code.colored_schedule(rng_seed=None)
    rebuilt = _degree_two_code().colored_schedule()

    assert first == second == explicit_none == rebuilt
    x_layers, z_layers = first
    _assert_optimal_edge_coloring(code.hx, x_layers)
    _assert_optimal_edge_coloring(code.hz, z_layers)
    assert len(x_layers) == len(z_layers) == 2


def test_seeded_colored_schedule_is_reproducible_varied_and_optimal():
    code = _degree_two_code()
    schedules = {}

    for seed in range(16):
        schedule = code.colored_schedule(rng_seed=seed)
        assert schedule == code.colored_schedule(rng_seed=seed)
        assert schedule == _degree_two_code().colored_schedule(rng_seed=seed)
        x_layers, z_layers = schedule
        _assert_optimal_edge_coloring(code.hx, x_layers)
        _assert_optimal_edge_coloring(code.hz, z_layers)
        schedules[schedule] = seed

    assert len(schedules) > 1
