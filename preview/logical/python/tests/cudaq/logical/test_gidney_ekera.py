# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_ROOT = ROOT / "examples"


@pytest.fixture(scope="module")
def factory():
    sys.path.insert(0, str(EXAMPLE_ROOT))
    try:
        yield importlib.import_module("gidney_ekera_factory")
    finally:
        sys.path.remove(str(EXAMPLE_ROOT))


def test_rsa2048_operating_point_reproduces_paper_layout(factory):
    point = factory.OPERATING_POINTS[2048]
    assert point is factory.RSA_2048
    assert point.exponent_qubits == 3_029
    assert point.padding == 38
    assert point.carry_pieces == 2
    assert point.piece_length == 1_062
    assert point.accumulator_width == 2_124
    assert point.address_width == 10
    assert point.table_rows == 1_023
    assert point.fixup_count == 64
    assert point.lookup_count == 505_965
    assert point.toffolis_per_lookup == 5_333
    assert point.level_1_lanes == 6
    assert (point.factory_box_width, point.factory_box_height) == (13, 7)
    assert point.factory_lanes == 28
    assert (point.piece_width, point.piece_height) == (99, 62)
    assert point.board_patches == 12_276
    assert point.factory_patches == 2_548
    assert point.compute_patches == 9_716


def test_factory_is_built_once_per_distance_pair(factory):
    rsa2048 = factory.factory_for(factory.RSA_2048)
    assert rsa2048 is factory.surface_autoccz_factory(15, 27)
    assert rsa2048.level_1_code.d.conservative_value == 15
    assert rsa2048.code.d.conservative_value == 27

    other = factory.surface_autoccz_factory(17, 29)
    assert other is not rsa2048
    assert other.code.d.conservative_value == 29
    assert other.autoccz_factory is not rsa2048.autoccz_factory


def test_rsa2048_factory_characterization_matches_reference(factory):
    point = factory.RSA_2048
    estimate, model = factory.characterize_surface_autoccz_factory(point)
    assert estimate.event_count == point.factory_schedule_events
    assert model.startup_cycles == pytest.approx(point.factory_startup_cycles)
    assert model.output_interval_cycles == pytest.approx(
        point.factory_output_interval_cycles)
    assert (model.characterization.physical_units ==
            point.factory_lane_physical_qubits)
