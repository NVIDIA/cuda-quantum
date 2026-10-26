# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Success-predicate constructors for ``GadgetProfile(success=...)`` rows.

``cudaq.logical.gadgets.all_zero(records)`` and
``cudaq.logical.gadgets.all_false(records)``
normalize record selections into the tuple of :class:`SuccessPredicate` rows
that the ``success=`` parameter of :class:`cudaq.logical.GadgetProfile` (and
the gadget profile builder accepts.
``cudaq.logical.gadgets.accept_all`` is the empty
selection: every shot is accepted.
"""

from __future__ import annotations

from cudaq.logical.gadgets.records import (
    InputSyndromeRef,
    ProfileParity,
    RecordFamily,
    RecordParity,
    RecordRef,
    RecordVectorParity,
    StructuredRecord,
)
from cudaq.logical.gadgets.semantics import SuccessPredicate

#: Accept every shot: the empty success-predicate row set.
accept_all: tuple[SuccessPredicate, ...] = ()


def _scalar_rows(records):
    if isinstance(records, RecordVectorParity):
        return records.rows()
    if isinstance(records, RecordFamily):
        return tuple(records[index] for index in range(len(records.indices)))
    if isinstance(records, StructuredRecord):
        if records.check_count:
            family = records.checks
        elif records.data_count:
            family = records.data
        elif records.bit_count:
            family = records.bits
        else:
            raise ValueError(
                "structured record selects no bits for a success row")
        return tuple(family[index] for index in range(len(family.indices)))
    if isinstance(records,
                  (RecordRef, RecordParity, ProfileParity, InputSyndromeRef)):
        return (records,)
    if isinstance(records, (tuple, list)):
        rows = []
        for item in records:
            rows.extend(_scalar_rows(item))
        return tuple(rows)
    raise TypeError(
        "success predicates require gadget records, record families/parities, "
        "or a sequence of them")


def all_zero(records):
    """Success rows requiring every selected record bit to be zero.

    Presentation annotations and execution policy are separate profile and
    protocol facets, respectively.
    """
    rows = _scalar_rows(records)
    if not rows:
        raise ValueError(
            "cudaq.logical.gadgets.all_zero requires at least one record")
    return tuple(SuccessPredicate(row) for row in rows)


def all_false(records):
    """Success rows requiring every selected flag/event record to be unset.

    Identical row semantics to :func:`all_zero`; the name matches flag-style
    records the way ``cudaq.logical.all_false`` matches traced events.
    """
    return all_zero(records)


__all__ = ["accept_all", "all_zero", "all_false"]
