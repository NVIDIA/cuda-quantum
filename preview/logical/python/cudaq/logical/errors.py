# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed CUDA-Q Logical user-surface exception hierarchy.

This module is an import leaf: it must not import other ``cudaq_logical`` modules at
module scope.
"""

from __future__ import annotations


class NoActiveTrace(RuntimeError):
    """A trace-only operation was invoked with no active CUDA-Q Logical trace."""


class CrossContextValue(ValueError):
    """A frontend value belongs to a different tracing builder or domain."""


class InvalidReversibleCall(ValueError):
    """A reversible helper call violates an operand or ancilla contract."""


class InvalidReversibleSignature(TypeError):
    """A reversible helper call has an invalid handle, policy, or collection."""


class AmbiguousLogicalPortMap(ValueError):
    """Several logical-port embeddings satisfy the declared objective."""


class ObjectiveMismatch(ValueError):
    """A realization does not implement its declared logical objective."""


class PlacementInfeasible(ValueError):
    """No placement satisfies the declared constraints and capacities."""


class MissingObjective(TypeError):
    """A gadget/protocol definition lacks a resolvable ``implements=``."""


class InvalidDefinitionSignature(TypeError):
    """A decorated CUDA-Q Logical definition has an invalid typed Python boundary."""


class EstimateOnlyTargetError(ValueError):
    """A logical demand surrogate was passed to an execution target."""


class UnsupportedCombination(TypeError):
    """No launch capability of the target can consume this build."""


class UnsupportedProfile(ValueError):
    """A build/definition profile is outside the operation's accepted set."""


class RepeatCountOverflow(OverflowError):
    """A concrete repeat count exceeds the signed 64-bit model domain."""


class InvalidPortBinding(ValueError):
    """An explicit logical-port binding does not match the derived action."""


class InvalidSyndromeSchedule(ValueError):
    """A code-specific extraction schedule is malformed or ambiguous."""


class InvalidCodeAlgebra(ValueError):
    """A QEC code's authored algebra is not in canonical reduced form."""


class UnserializableCapture(TypeError):
    """A decorated definition captured a value with no immutable snapshot."""


def __getattr__(name: str):
    # lazy re-export of ``UseAfterConsume``, ``LinkageError``, and ``NonCliffordAction`` using match-case.
    match name:
        case "UseAfterConsume":
            from .types.values import UseAfterConsume
            err = UseAfterConsume
        case "LinkageError":
            from .compiler.link_check import LinkageError
            err = LinkageError
        case "NonCliffordAction":
            from .algebra.clifford import NonCliffordAction
            err = NonCliffordAction
        case _:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = err
    return err


__all__ = [
    "NoActiveTrace",
    "CrossContextValue",
    "InvalidReversibleCall",
    "InvalidReversibleSignature",
    "AmbiguousLogicalPortMap",
    "ObjectiveMismatch",
    "PlacementInfeasible",
    "MissingObjective",
    "InvalidDefinitionSignature",
    "EstimateOnlyTargetError",
    "UnsupportedCombination",
    "UnsupportedProfile",
    "RepeatCountOverflow",
    "InvalidPortBinding",
    "InvalidSyndromeSchedule",
    "InvalidCodeAlgebra",
    "UnserializableCapture",
    "UseAfterConsume",
    "LinkageError",
    "NonCliffordAction",
]
