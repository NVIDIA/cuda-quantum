# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed QLX user-surface exception hierarchy.

The frontend and compiler raise typed diagnostics (spec 03 SS10) so callers can
select on the model rule rather than on message text.  Every class below also
subclasses the builtin exception it historically replaced, so existing
``except ValueError`` / ``pytest.raises(RuntimeError)`` call sites keep
working while new code catches the typed name or :class:`QLXError`.

This module is an import leaf: it must not import other ``qlx`` modules at
module scope.
"""

from __future__ import annotations


class QLXError(Exception):
    """Base class of every typed QLX user-surface diagnostic."""


class NoActiveTrace(QLXError, RuntimeError):
    """A trace-only operation was invoked with no active QLX trace."""


class CrossContextValue(QLXError, ValueError):
    """A frontend value belongs to a different tracing builder or domain."""


class InvalidReversibleCall(QLXError, ValueError):
    """A reversible helper call violates an operand or ancilla contract."""


class InvalidReversibleSignature(QLXError, TypeError):
    """A reversible helper call has an invalid handle, policy, or collection."""


class AmbiguousLogicalPortMap(QLXError, ValueError):
    """Several logical-port embeddings satisfy the declared objective."""


class ObjectiveMismatch(QLXError, ValueError):
    """A realization does not implement its declared logical objective."""


class PlacementInfeasible(QLXError, ValueError):
    """No placement satisfies the declared constraints and capacities."""


class MissingObjective(QLXError, TypeError):
    """A gadget/protocol definition lacks a resolvable ``implements=``."""


class InvalidDefinitionSignature(QLXError, TypeError):
    """A decorated QLX definition has an invalid typed Python boundary."""


class NotSampleable(QLXError, NotImplementedError):
    """An explicit native-sample policy was requested for a feedback-bearing
    P3 program, which can only be inspected, scheduled, estimated, or emitted."""


class SamplingCapacityError(QLXError, MemoryError):
    """An exact dense sampling request exceeds its declared memory bound."""


class EstimateOnlyTargetError(QLXError, ValueError):
    """A logical demand surrogate was passed to an execution target."""


class UnsupportedCombination(QLXError, TypeError):
    """No launch capability of the target can consume this build."""


class UnsupportedProfile(QLXError, ValueError):
    """A build/definition profile is outside the operation's accepted set."""


class RepeatCountOverflow(QLXError, OverflowError):
    """A concrete repeat count exceeds the signed 64-bit model domain."""


class InvalidPortBinding(QLXError, ValueError):
    """An explicit logical-port binding does not match the derived action."""


class InvalidSyndromeSchedule(QLXError, ValueError):
    """A code-specific extraction schedule is malformed or ambiguous."""


class InvalidCodeAlgebra(QLXError, ValueError):
    """A QEC code's authored algebra is not in canonical reduced form."""


class UnserializableCapture(QLXError, TypeError):
    """A decorated definition closed over a value that cannot be snapshotted."""


class IncompletePhysicalModel(QLXError, ValueError):
    """A physical stage requires a supply a device does not physically model.

    A resource stream that is neither backed by a factory region nor marked
    ``external=True`` has no physical home; P3 lowering and schedule-aware or
    digital-twin estimation fail closed on it rather than pricing its supply
    at zero."""


class PhysicalProjectionCapacityError(QLXError, ValueError):
    """P2-to-P3 projection exhausted a host implementation resource.

    The retained Fabric composition remains semantically valid: this typed
    diagnostic reports that the current Python projector cannot traverse it
    within the host recursion capacity.  It is deliberately also a
    :class:`ValueError` for compatibility with the projection boundary's
    historical failure contract.
    """


class UnsupportedSchedulingStrategy(QLXError, ValueError):
    """A physical scheduling request names no executable strategy provider.

    The diagnostic is raised before P3 prerequisite compilation or schedule
    production.  Callers can therefore distinguish an unavailable/unknown
    scheduling request from an infeasible physical event graph.
    """


class ScheduleConflict(QLXError, ValueError):
    """A scheduling request or detached schedule conflicts with verified P3 IR.

    Physical schedules are immutable projections of one retained
    ``phys.schedule`` operation.  Rescheduling, replacing their graph/device
    context, or supplying divergent detached rows fails before any consumer or
    compiler pass can use the conflicting values.
    """


def __getattr__(name: str):
    # ``UseAfterConsume`` is defined next to the linear value proxies and
    # already subclasses ``QLXError``; re-export it lazily to keep this module
    # an import leaf.
    if name == "UseAfterConsume":
        from cudaq.logical.types.values import UseAfterConsume

        return UseAfterConsume
    owners = {
        "LinkageError": ("cudaq.logical.compiler.link_check", "LinkageError"),
        "NonCliffordAction":
            ("cudaq.logical.algebra.clifford", "NonCliffordAction"),
        "ProfileSemanticError":
            ("cudaq.logical.gadgets", "ProfileSemanticError"),
        "UnavailableTargetError":
            ("cudaq.logical.targets", "UnavailableTargetError"),
        "MissingEvidence": ("cudaq.logical.estimate.types", "MissingEvidence"),
    }
    owner = owners.get(name)
    if owner is not None:
        from importlib import import_module

        value = getattr(import_module(owner[0]), owner[1])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QLXError",
    "NoActiveTrace",
    "CrossContextValue",
    "InvalidReversibleCall",
    "InvalidReversibleSignature",
    "AmbiguousLogicalPortMap",
    "ObjectiveMismatch",
    "PlacementInfeasible",
    "MissingObjective",
    "InvalidDefinitionSignature",
    "NotSampleable",
    "SamplingCapacityError",
    "EstimateOnlyTargetError",
    "UnsupportedCombination",
    "UnsupportedProfile",
    "RepeatCountOverflow",
    "InvalidPortBinding",
    "InvalidSyndromeSchedule",
    "InvalidCodeAlgebra",
    "UnserializableCapture",
    "IncompletePhysicalModel",
    "PhysicalProjectionCapacityError",
    "UnsupportedSchedulingStrategy",
    "ScheduleConflict",
    "UseAfterConsume",
    "LinkageError",
    "NonCliffordAction",
    "ProfileSemanticError",
    "UnavailableTargetError",
    "MissingEvidence",
]
