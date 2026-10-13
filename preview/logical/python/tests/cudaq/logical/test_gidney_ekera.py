# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import importlib
import runpy
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


@pytest.fixture(scope="module")
def example(factory):
    return runpy.run_path(str(EXAMPLE_ROOT / "05_gidney_ekera.py"),
                          run_name="gidney_ekera_example")


@pytest.fixture(autouse=True)
def reset_cudaq_target_after_test():
    yield
    import cudaq

    cudaq.reset_target()


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


@pytest.mark.parametrize(
    "modulus_bits, expected",
    (
        (3072,
         dict(exponent_qubits=4_565,
              padding=42,
              carry_pieces=3,
              piece_length=1_066,
              address_width=9,
              fixup_count=46,
              lookup_count=1_422_454,
              toffolis_per_lookup=6_950,
              factory_box_width=14,
              factory_box_height=8,
              factory_lanes=48,
              piece_width=121,
              piece_height=58)),
        (4096,
         dict(exponent_qubits=6_101,
              padding=46,
              carry_pieces=4,
              piece_length=1_070,
              address_width=9,
              fixup_count=46,
              lookup_count=2_528_255,
              toffolis_per_lookup=9_113,
              factory_box_width=13,
              factory_box_height=7,
              factory_lanes=64,
              piece_width=113,
              piece_height=59)),
    ),
)
def test_larger_operating_points_derive_paper_layout(factory, modulus_bits,
                                                     expected):
    point = factory.OPERATING_POINTS[modulus_bits]
    assert {name: getattr(point, name) for name in expected} == expected


def test_factory_is_built_once_per_distance_pair(factory):
    rsa2048 = factory.factory_for(factory.RSA_2048)
    assert rsa2048 is factory.surface_autoccz_factory(15, 27)
    assert rsa2048.level_1_code.d.conservative_value == 15
    assert rsa2048.code.d.conservative_value == 27

    other = factory.surface_autoccz_factory(17, 29)
    assert other is not rsa2048
    assert other.code.d.conservative_value == 29
    assert other.autoccz_factory is not rsa2048.autoccz_factory


@pytest.mark.parametrize("modulus_bits", (2048, 3072, 4096))
def test_factory_characterization_matches_reference(factory, modulus_bits):
    point = factory.OPERATING_POINTS[modulus_bits]
    estimate, model = factory.characterize_surface_autoccz_factory(point)
    assert estimate.event_count == point.factory_schedule_events
    assert model.startup_cycles == pytest.approx(point.factory_startup_cycles)
    assert model.output_interval_cycles == pytest.approx(
        point.factory_output_interval_cycles)
    assert (model.characterization.physical_units ==
            point.factory_lane_physical_qubits)


# Table 3 reports `megaqubits` and hours rounded up to two significant digits.
@pytest.mark.parametrize("modulus_bits, megaqubits, hours", (
    (2048, 20, 5.1),
    (3072, 38, 12),
    (4096, 55, 22),
))
def test_resource_kernel_matches_operating_point(factory, example, modulus_bits,
                                                 megaqubits, hours):
    import cudaq
    import cudaq.logical as cql

    point = factory.OPERATING_POINTS[modulus_bits]
    cudaq.set_target(cql.targets.estimator)
    estimate = cudaq.estimate(example["build_resource_kernel"](point))
    logical = cql.estimate.LogicalEstimate.from_annotations(
        estimate.annotations)

    # Accumulator, bus, address, runways, two ancillas, unlookup qubit.
    assert logical.logical_qubits_peak == (2 * point.accumulator_width +
                                           point.address_width +
                                           point.carry_pieces + 3)
    assert logical.synthesis_demand == {
        "qlx_standard_ccx": point.lookup_count * point.toffolis_per_lookup
    }

    result = example["calculate_analytical_metrics"](logical, point)
    assert result.folded_lookups == point.lookup_count
    assert 0.95 * megaqubits <= result.physical_qubits / 1e6 <= megaqubits
    assert 0.9 * hours <= result.runtime_hours <= hours
