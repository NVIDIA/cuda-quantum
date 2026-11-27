# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Optional Stim interoperability for independent detector error models."""

import cudaq_edem as edem


def to_stim_dem(model: edem.EDEM | edem.DEM):
    """Format a closed independent model without changing its columns or laws.

    Run the explicit factorization and column coalescing operations first.
    This optional adapter only translates row IDs and writes instructions.
    """
    import stim

    if not model.is_dem:
        raise ValueError("Stim export requires closed code boundaries")
    if any(model.successes):
        raise ValueError("Stim DEM export does not support postselection")
    if any(group.kind != edem.GroupKind.INDEPENDENT
           for fault in model.faults
           for group in fault.groups):
        raise ValueError("run edem.factorize_dem before Stim formatting")
    offsets, targets, start = [], [], 0
    for fault in model.faults:
        offsets.append(start)
        targets.extend([] for _ in range(fault.size))
        start += fault.size
    row_offsets = {}
    for kind, sizes in ((edem.PortKind.DETECTOR, model.detectors),
                        (edem.PortKind.OBSERVABLE, model.observables)):
        start = 0
        for index, size in enumerate(sizes):
            row_offsets[kind, index] = start
            start += size
    for block in model.blocks:
        for row, column in block.matrix.entries:
            index = row_offsets[block.output.kind, block.output.index] + row
            target = (stim.target_relative_detector_id(index)
                      if block.output.kind == edem.PortKind.DETECTOR else
                      stim.target_logical_observable_id(index))
            targets[offsets[block.input.index] + column].append(target)
    result = stim.DetectorErrorModel()
    for offset, fault in zip(offsets, model.faults):
        for group in fault.groups:
            for column, probability in zip(group.fault_ids,
                                           group.probabilities):
                result.append(
                    stim.DemInstruction("error", [probability],
                                        targets[offset + column]))
    for row in range(sum(model.detectors)):
        result.append("detector", [], [stim.target_relative_detector_id(row)])
    for row in range(sum(model.observables)):
        result.append("logical_observable", [],
                      [stim.target_logical_observable_id(row)])
    return result


__all__ = ["to_stim_dem"]
