# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pytest

from cudaq.logical.algebra.pauli import PauliFactor, PauliProduct
from cudaq.logical.architecture.logical import Space
from cudaq.logical.codes import QECBlockOwner
from cudaq.logical.qec.lattice_surgery import (
    ProductMeasurement,
    QECNetworkAction,
    QECNetworkArtifact,
    QECNetworkEpoch,
    QECNetworkPlan,
    QECNetworkRegion,
    QECNetworkRequest,
    validate_network_plan,
)
from cudaq.logical.qec.lowering import ActionSiteHandle


def _action(
        name: str,
        logical_indices: tuple[int, ...],
        *,
        blocks: tuple[str, ...],
        paulis: str | tuple[str, ...] = "X",
        canonical_owner_order: tuple[int, ...] | None = None,
        after: Iterable[ProductMeasurement] = (),
) -> QECNetworkAction:
    if isinstance(paulis, str):
        paulis = (paulis,) * len(logical_indices)
    if len(paulis) != len(logical_indices):
        raise ValueError("test Pauli labels must match logical indices")
    if canonical_owner_order is None:
        canonical_owner_order = tuple(range(len(logical_indices)))
    if sorted(canonical_owner_order) != list(range(len(logical_indices))):
        raise ValueError("canonical owner order must be a permutation")
    # Repeated actions on one encoded port refer to the same logical slot.
    spaces = {
        block: Space(name=f"{block}_space", capacity=max(logical_indices) + 1)
        for block in set(blocks)
    }
    slots = tuple(spaces[block][logical_index]
                  for block, logical_index in zip(blocks, logical_indices))
    product = PauliProduct(
        tuple(
            PauliFactor(slots[owner], paulis[owner])
            for owner in canonical_owner_order))
    measurement = ProductMeasurement(product, name=name, after=after)
    placements = tuple(
        f"{name}.owner{index}" for index in range(len(logical_indices)))
    site = ActionSiteHandle(
        symbol=name,
        kind="objective",
        objective_family="pauli_product_measurement",
        objective="test.logical_product",
        placements=placements,
        parameters={
            "x_mask": sum(
                1 << index for index, pauli in enumerate(paulis)
                if pauli in {"X", "Y"}),
            "z_mask": sum(
                1 << index for index, pauli in enumerate(paulis)
                if pauli in {"Z", "Y"}),
            "sign": 1,
        },
        input_arity=len(logical_indices),
        result_arity=len(logical_indices) + 1,
    )
    return QECNetworkAction(
        site=site,
        measurement=measurement,
        owners=tuple(
            QECBlockOwner(placement=placement, logical_index=logical_index)
            for placement, logical_index in zip(placements, logical_indices)),
        blocks=blocks,
        after=measurement.after,
        region="region0",
    )


def _request_and_plan(
    actions: tuple[QECNetworkAction, ...],
    *,
    block_sharing: str,
) -> tuple[QECNetworkRequest, QECNetworkPlan]:
    names = tuple(action.site.symbol for action in actions)
    request = QECNetworkRequest(
        source_sha256="0" * 64,
        lowering_manifest_sha256="1" * 64,
        device_architecture_sha256="2" * 64,
        actions=actions,
        regions=(QECNetworkRegion(
            id="region0",
            actions=names,
            live_inputs=(),
            live_outputs=(),
        ),),
    )
    plan = QECNetworkPlan(
        request_sha256=request.digest,
        lowering_manifest_sha256=request.lowering_manifest_sha256,
        device_architecture_sha256=request.device_architecture_sha256,
        policy_sha256=request.policy_sha256,
        provider_key="test.block_sharing",
        required_projector_key="test.block_sharing",
        required_projector_pipeline_sha256="3" * 64,
        epochs=(QECNetworkEpoch(
            id="epoch0",
            region="region0",
            actions=names,
            block_sharing=block_sharing,
        ),),
        artifact=QECNetworkArtifact(
            schema="test.block_sharing/v1",
            media_type="application/json",
            payload={},
        ),
    )
    return request, plan


def test_exclusive_epoch_preserves_legacy_serialization_and_replay() -> None:
    epoch = QECNetworkEpoch(
        id="epoch0",
        region="region0",
        actions=("measure0",),
    )
    legacy = {
        "id": "epoch0",
        "region": "region0",
        "actions": ["measure0"],
        "claims": [],
    }
    assert epoch.block_sharing == "exclusive"
    assert epoch.to_dict() == legacy
    assert QECNetworkEpoch.from_dict(legacy) == epoch

    with pytest.raises(ValueError, match="fields must be exactly"):
        QECNetworkEpoch.from_dict({**legacy, "unknown": True})
    with pytest.raises(ValueError, match="block_sharing"):
        QECNetworkEpoch(
            id="epoch0",
            region="region0",
            actions=("measure0",),
            block_sharing="shared",
        )


def test_commuting_disjoint_products_epoch_round_trips_explicitly() -> None:
    epoch = QECNetworkEpoch(
        id="epoch0",
        region="region0",
        actions=("measure0", "measure1"),
        block_sharing="commuting_disjoint_products",
    )
    payload = epoch.to_dict()
    assert payload["block_sharing"] == "commuting_disjoint_products"
    assert QECNetworkEpoch.from_dict(payload) == epoch


def test_same_block_disjoint_logical_products_can_share_an_epoch() -> None:
    left = _action("measure01", (0, 1), blocks=("block0", "block0"))
    right = _action("measure23", (2, 3), blocks=("block0", "block0"))
    request, plan = _request_and_plan(
        (left, right),
        block_sharing="commuting_disjoint_products",
    )

    validate_network_plan(request, plan)

    with pytest.raises(ValueError, match="reuses one encoded block"):
        validate_network_plan(
            request,
            _request_and_plan((left, right), block_sharing="exclusive")[1],
        )


def test_shared_block_batch_rejects_an_overlapping_logical_port() -> None:
    left = _action("measure01", (0, 1), blocks=("block0", "block0"))
    overlap = _action("measure12", (1, 2), blocks=("block0", "block0"))
    request, plan = _request_and_plan(
        (left, overlap),
        block_sharing="commuting_disjoint_products",
    )

    with pytest.raises(ValueError, match="overlap a logical port"):
        validate_network_plan(request, plan)


def test_shared_block_batch_rejects_multiple_encoded_blocks() -> None:
    left = _action("measure01", (0, 1), blocks=("block0", "block0"))
    right = _action("measure23", (2, 3), blocks=("block1", "block1"))
    request, plan = _request_and_plan(
        (left, right),
        block_sharing="commuting_disjoint_products",
    )

    with pytest.raises(ValueError, match="share the same encoded block"):
        validate_network_plan(request, plan)


def test_shared_block_batch_requires_multiple_actions() -> None:
    action = _action("measure01", (0, 1), blocks=("block0", "block0"))
    request, plan = _request_and_plan(
        (action,),
        block_sharing="commuting_disjoint_products",
    )

    with pytest.raises(ValueError, match="at least two actions"):
        validate_network_plan(request, plan)


def test_shared_block_batch_cannot_co_schedule_a_dependency() -> None:
    first = _action("measure01", (0, 1), blocks=("block0", "block0"))
    dependent = _action(
        "measure23",
        (2, 3),
        blocks=("block0", "block0"),
        after=(first.measurement,),
    )
    request, plan = _request_and_plan(
        (first, dependent),
        block_sharing="commuting_disjoint_products",
    )

    with pytest.raises(ValueError, match="co-schedules an action dependency"):
        validate_network_plan(request, plan)


def test_commuting_products_epoch_round_trips_explicitly() -> None:
    epoch = QECNetworkEpoch(
        id="epoch0",
        region="region0",
        actions=("measure0", "measure1"),
        block_sharing="commuting_products",
    )
    payload = epoch.to_dict()
    assert payload["block_sharing"] == "commuting_products"
    assert QECNetworkEpoch.from_dict(payload) == epoch


def test_commuting_products_allows_overlapping_adjacent_zz_products() -> None:
    left = _action(
        "measure01",
        (0, 1),
        blocks=("block0", "block0"),
        paulis="Z",
    )
    right = _action(
        "measure12",
        (1, 2),
        blocks=("block0", "block0"),
        paulis="Z",
    )
    request, plan = _request_and_plan(
        (left, right),
        block_sharing="commuting_products",
    )

    validate_network_plan(request, plan)


def test_commuting_products_allows_internal_dependencies() -> None:
    first = _action(
        "measure01",
        (0, 1),
        blocks=("block0", "block0"),
        paulis="Z",
    )
    dependent = _action(
        "measure12",
        (1, 2),
        blocks=("block0", "block0"),
        paulis="Z",
        after=(first.measurement,),
    )
    request, plan = _request_and_plan(
        (first, dependent),
        block_sharing="commuting_products",
    )

    validate_network_plan(request, plan)

    reversed_epoch = replace(plan.epochs[0],
                             actions=(dependent.site.symbol, first.site.symbol))
    with pytest.raises(ValueError, match="reverses or co-schedules"):
        validate_network_plan(request, replace(plan, epochs=(reversed_epoch,)))


def test_commuting_products_rejects_anticommuting_overlap() -> None:
    left = _action(
        "measure01",
        (0, 1),
        blocks=("block0", "block0"),
        paulis="Z",
    )
    right = _action(
        "measure12",
        (1, 2),
        blocks=("block0", "block0"),
        paulis="X",
    )
    request, plan = _request_and_plan(
        (left, right),
        block_sharing="commuting_products",
    )

    with pytest.raises(ValueError, match="anticommute"):
        validate_network_plan(request, plan)


def test_commuting_products_uses_measurements_even_if_site_masks_are_swapped(
        ) -> None:
    mixed = _action(
        "measure01_mixed",
        (0, 1),
        blocks=("block0", "block0"),
        paulis=("X", "Z"),
    )
    overlap = _action(
        "measure0_z",
        (0,),
        blocks=("block0",),
        paulis="Z",
    )
    # The multiset of mask letters still matches, but the assignment to
    # owners no longer matches the actual X0 Z1 measurement.
    swapped = replace(mixed, site=replace(
        mixed.site,
        parameters={"x_mask": 2, "z_mask": 1, "sign": 1},
    ))
    request, plan = _request_and_plan(
        (swapped, overlap),
        block_sharing="commuting_products",
    )
    with pytest.raises(ValueError, match="anticommute"):
        validate_network_plan(request, plan)


def test_commuting_products_accepts_two_anticommutations() -> None:
    left = _action(
        "measure01_z",
        (0, 1),
        blocks=("block0", "block0"),
        paulis="Z",
    )
    right = _action(
        "measure01_x",
        (0, 1),
        blocks=("block0", "block0"),
        paulis="X",
    )
    request, plan = _request_and_plan(
        (left, right),
        block_sharing="commuting_products",
    )

    validate_network_plan(request, plan)


@pytest.mark.parametrize(("overlap_pauli", "commutes"), (
    ("X", True),
    ("Z", False),
))
def test_commuting_products_uses_slot_identity_when_factors_are_reordered(
        overlap_pauli: str, commutes: bool) -> None:
    mixed = _action(
        "measure01_mixed",
        (0, 1),
        blocks=("block0", "block0"),
        paulis=("X", "Z"),
        canonical_owner_order=(1, 0),
    )
    assert tuple(term.pauli for term in mixed.measurement.terms) == ("X", "Z")
    overlap = _action(
        "measure0_overlap",
        (0,),
        blocks=("block0",),
        paulis=overlap_pauli,
    )
    request, plan = _request_and_plan(
        (mixed, overlap),
        block_sharing="commuting_products",
    )

    if commutes:
        validate_network_plan(request, plan)
    else:
        with pytest.raises(ValueError, match="anticommute"):
            validate_network_plan(request, plan)
