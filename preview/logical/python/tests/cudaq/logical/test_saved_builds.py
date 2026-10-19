# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import pytest

import cudaq
import cudaq.logical as cql


@cudaq.kernel
def readout():
    qubits = cudaq.qvector(1)
    mz(qubits[0])


@pytest.fixture
def physical_target():
    target = cql.targets.surface_physical_target(logical_capacity=1)
    cudaq.set_target(target)
    yield target
    cudaq.reset_target()


@pytest.fixture
def estimator_target():
    cudaq.set_target(cql.targets.estimator)
    yield cql.targets.estimator
    cudaq.reset_target()


STAGES = {"p0", "clifford_t", "p1", "p2", "p3"}


def test_estimate_saves_every_build_of_the_stack(physical_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    saved = result.annotations["BUILDS"]
    assert set(saved) == STAGES
    for stage, files in saved.items():
        assert set(files) == {"mlir", "build"}
        for path in files.values():
            assert path.startswith(str(tmp_path))
            assert (tmp_path / path.rsplit("/", 1)[1]).is_file()
        assert f".{stage}." in files["mlir"]
    # Everything but the builds is reported as usual.
    assert {"LOGICAL", "STATIC", "ANALYTICAL",
            "SCHEDULE"} <= set(result.annotations)


def test_estimate_saves_nothing_unless_asked(physical_target, tmp_path):
    result = cudaq.estimate(readout)

    assert "BUILDS" not in result.annotations
    assert list(tmp_path.iterdir()) == []


def test_saved_files_are_named_after_the_kernel_stage_and_digest(
        physical_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    for stage in STAGES:
        build = cql.load_build(result, stage=stage)
        digest = build.content_sha256.removeprefix("sha256:")[:12]
        name = result.annotations["BUILDS"][stage]["mlir"].rsplit("/", 1)[1]
        assert name == f"readout.{stage}.{digest}.mlir"


def test_saved_mlir_is_the_build_mlir(physical_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    for stage, files in result.annotations["BUILDS"].items():
        text = open(files["mlir"]).read()
        assert text == cql.load_build(result, stage=stage).to_mlir()


@pytest.mark.parametrize("stage,profile", [("p0", "p0"), ("p1", "p1"),
                                           ("p2", "p2n"), ("p3", "p3")])
def test_load_build_returns_the_stage_build(physical_target, tmp_path, stage,
                                            profile):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    build = cql.load_build(result, stage=stage)

    assert build.profile == profile


def test_load_build_defaults_to_the_last_stage(physical_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    assert (cql.load_build(result).content_sha256 == cql.load_build(
        result, stage="p3").content_sha256)


def test_loaded_physical_build_estimates_like_the_estimate(
        physical_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    schedule = cql.compiler.schedule(cql.load_build(result))
    # A loaded build does not retain the device, so its calibration must be
    # given.
    estimate = cql.estimate(schedule,
                            tier=cql.estimate.Tier.SCHEDULE,
                            p_phys=1.0e-3,
                            failure_budget=0.1,
                            cycle_time=1.0e-9)

    reported = result.annotations["SCHEDULE"]
    assert estimate.physical_qubits == reported["physical_qubits"]
    assert estimate.event_count == reported["event_count"]


def test_load_build_accepts_a_build_file(estimator_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    build = cql.load_build(result.annotations["BUILDS"]["p0"]["build"])

    assert build.profile == "p0"
    assert "qlx.measure" in build.to_mlir()


def test_load_build_detects_a_modified_build_file(estimator_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)
    path = tmp_path / result.annotations["BUILDS"]["p0"]["build"].rsplit(
        "/", 1)[1]
    path.write_bytes(path.read_bytes().replace(b"qlx.measure", b"qlx.measurf"))

    with pytest.raises(ValueError):
        cql.load_build(path)


def test_estimate_creates_the_directory(estimator_target, tmp_path):
    directory = tmp_path / "nested" / "builds"

    cudaq.estimate(readout, save_temps=directory)

    assert any(directory.glob("readout.p0.*.build"))


def test_a_stack_without_a_stage_saves_no_build_for_it(estimator_target,
                                                       tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    assert set(result.annotations["BUILDS"]) == {"p0"}


def test_load_build_rejects_an_estimate_without_saved_builds(estimator_target):
    result = cudaq.estimate(readout)

    with pytest.raises(ValueError, match="save_temps"):
        cql.load_build(result)


def test_load_build_rejects_an_unsaved_stage(estimator_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)

    with pytest.raises(ValueError, match="p3.*saved: p0"):
        cql.load_build(result, stage="p3")


def test_load_build_rejects_a_stage_with_a_path(estimator_target, tmp_path):
    result = cudaq.estimate(readout, save_temps=tmp_path)
    path = result.annotations["BUILDS"]["p0"]["build"]

    with pytest.raises(TypeError, match="stage"):
        cql.load_build(path, stage="p0")


def test_load_build_rejects_other_sources():
    with pytest.raises(TypeError, match="EstimateResult"):
        cql.load_build(42)
