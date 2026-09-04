# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
`integrator_options` must be per-instance. `BaseIntegrator.__init__` used to
mutate a class-level dict via `.update(kwargs)` instead of assigning a fresh
one, so two integrators created in the same process shared one options dict
and silently inherited each other's settings. This does not need a GPU or a
dynamics target: it is a plain object-construction bug.
"""
import pytest

pytest.importorskip("cupy")

from cudaq.dynamics.integrators.builtin_integrators import RungeKuttaIntegrator


def test_integrator_options_not_shared_across_instances():
    first = RungeKuttaIntegrator(order=1, max_step_size=0.01)
    assert first.order == 1
    assert first.max_step_size == 0.01

    # A second integrator that only overrides `order` must not pick up
    # `max_step_size` from the first one.
    second = RungeKuttaIntegrator(order=2)
    assert second.max_step_size is None
    assert second.integrator_options == {'order': 2}

    # The first integrator's options must also stay untouched by the second
    # construction.
    assert first.integrator_options == {'order': 1, 'max_step_size': 0.01}
