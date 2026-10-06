# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Resolve physical-estimation inputs from a selected device."""

from __future__ import annotations

from .types import Scaling


def _selected_device(root, explicit_device):
    if explicit_device is not None:
        return explicit_device
    build = getattr(root, "build", root)
    device = getattr(build, "_device", None)
    if device is not None:
        return device
    return getattr(root, "device", None)


def resolve_physical_parameters(
    root,
    *,
    device=None,
    p_phys=None,
    scaling=None,
    cycle_time=None,
):
    """Use explicit overrides, then the selected device's operating point.

    Physical error, surface-code scaling, and cycle duration describe a device
    operating point. Estimation callers may override them for sensitivity
    studies, but ordinary estimation should not have to restate values already
    captured by the compiled artifact's selected device.
    """

    selected = _selected_device(root, device)
    point = None if selected is None else selected.operating_point
    calibration = {} if point is None else point.calibration
    timing = {} if point is None else point.timing

    if p_phys is None:
        if "physical_error" not in calibration:
            raise TypeError(
                "p_phys= is required when the selected device operating "
                "point has no 'physical_error' calibration")
        p_phys = calibration["physical_error"]

    if scaling is None:
        has_prefactor = "surface_scaling_prefactor" in calibration
        has_threshold = "surface_threshold" in calibration
        if has_prefactor != has_threshold:
            raise ValueError(
                "the selected device operating point must provide both "
                "'surface_scaling_prefactor' and 'surface_threshold'")
        scaling = (Scaling(
            prefactor=float(calibration["surface_scaling_prefactor"]),
            threshold=float(calibration["surface_threshold"]),
        ) if has_prefactor else Scaling())

    if cycle_time is None:
        for name in ("surface_cycle_ns", "cycle_ns"):
            if name in timing:
                cycle_time = float(timing[name]) * 1.0e-9
                break
        else:
            if point is not None:
                raise TypeError(
                    "cycle_time= is required when the selected device "
                    "operating point has no 'surface_cycle_ns' or 'cycle_ns' "
                    "timing")
            # Preserve the historical unit-cycle default when estimation has
            # no selected physical operating point.
            cycle_time = 1.0

    return p_phys, scaling, cycle_time
