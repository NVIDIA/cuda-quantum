# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared decorator-boundary validation for P2 callable definitions."""

from __future__ import annotations

from inspect import Parameter, Signature, signature
from types import MappingProxyType, NoneType
from typing import Any, Callable, Mapping, get_args, get_origin, get_type_hints

from ..errors import InvalidDefinitionSignature
from ..types.semantic import float64, index, record, resource

_GADGET_INPUT = (
    "cudaq.logical.patch[Code|Encoding], cudaq.logical.types.resource[ResourceKind], or "
    "cudaq.logical.types.record[Code|Encoding]")
_PROTOCOL_INPUT = f"{_GADGET_INPUT}, or bool"
_OUTPUT = ("None or a fixed tuple/list of cudaq.logical.patch[Code|Encoding], "
           "cudaq.logical.types.resource[ResourceKind], "
           "cudaq.logical.types.record[Code|Encoding], or bool")
_GADGET_SCALARS = frozenset((bool, int, float, index, float64))


def _slot_error(decorator: str, provider: Callable[..., Any], slot: str,
                category: str, detail: str, expected: str):
    problem = f"{category} {detail}" if detail else category
    return InvalidDefinitionSignature(
        f"{decorator} definition {provider.__qualname__!r} {slot} has "
        f"{problem}; expected {expected}")


def _resolve_slot(
    provider: Callable[..., Any],
    *,
    decorator: str,
    slot: str,
    annotation: Any,
    expected: str,
    localns: Mapping[str, Any],
):
    # Resolve one slot at a time through typing's public API.  A one-entry
    # holder retains get_type_hints' normal forward-reference behavior without
    # losing which parameter or return annotation failed.
    def annotation_holder():
        pass

    annotation_holder.__annotations__ = {"value": annotation}
    try:
        return get_type_hints(
            annotation_holder,
            globalns=provider.__globals__,
            localns=dict(localns),
        )["value"]
    except (AttributeError, NameError, SyntaxError, TypeError) as exc:
        raise _slot_error(
            decorator,
            provider,
            slot,
            "an unresolvable annotation",
            repr(annotation),
            expected,
        ) from exc


def _is_encoded_boundary(annotation: Any, patch_type: type) -> bool:
    origin = get_origin(annotation)
    if origin not in (patch_type, record):
        return False
    arguments = get_args(annotation)
    if len(arguments) != 1:
        return False
    from ..codes import Code, Encoding

    return isinstance(arguments[0], (Code, Encoding))


def _is_resource_boundary(annotation: Any) -> bool:
    if get_origin(annotation) is not resource:
        return False
    arguments = get_args(annotation)
    if len(arguments) != 1:
        return False
    from ..std import ResourceKind

    return isinstance(arguments[0], ResourceKind)


def _input_expectation(definition_kind: str, allow_gadget_scalars: bool) -> str:
    if definition_kind == "protocol":
        return _PROTOCOL_INPUT
    expected = _GADGET_INPUT
    if allow_gadget_scalars:
        expected += (
            ", or an explicit-spec scalar linked by GadgetSpec.parameter_map")
    return expected


def _validate_input(
    annotation: Any,
    *,
    decorator: str,
    provider: Callable[..., Any],
    name: str,
    definition_kind: str,
    patch_type: type,
    allow_gadget_scalars: bool,
) -> None:
    valid = (_is_encoded_boundary(annotation, patch_type) or
             _is_resource_boundary(annotation))
    if definition_kind == "protocol":
        valid = valid or annotation is bool
        expected = _input_expectation(definition_kind, allow_gadget_scalars)
    else:
        valid = valid or (allow_gadget_scalars and
                          annotation in _GADGET_SCALARS)
        expected = _input_expectation(definition_kind, allow_gadget_scalars)
    if not valid:
        raise _slot_error(
            decorator,
            provider,
            f"parameter {name!r}",
            "an incompatible annotation",
            repr(annotation),
            expected,
        )


def _validate_output(
    annotation: Any,
    *,
    decorator: str,
    provider: Callable[..., Any],
    patch_type: type,
    path: str = "return annotation",
) -> None:
    if annotation in (None, NoneType):
        return
    origin = get_origin(annotation)
    if origin in (tuple, list):
        if path != "return annotation":
            raise _slot_error(
                decorator,
                provider,
                path,
                "an unsupported nested container shape",
                repr(annotation),
                _OUTPUT,
            )
        arguments = get_args(annotation)
        if any(item is Ellipsis for item in arguments):
            raise _slot_error(
                decorator,
                provider,
                path,
                "an unsupported variable-length shape",
                repr(annotation),
                _OUTPUT,
            )
        for item_index, item in enumerate(arguments):
            _validate_output(
                item,
                decorator=decorator,
                provider=provider,
                patch_type=patch_type,
                path=f"{path} element [{item_index}]",
            )
        return
    if (annotation is bool or _is_encoded_boundary(annotation, patch_type) or
            _is_resource_boundary(annotation)):
        return
    raise _slot_error(
        decorator,
        provider,
        path,
        "an incompatible annotation",
        repr(annotation),
        _OUTPUT,
    )


def resolve_definition_signature(
    provider: Callable[..., Any],
    *,
    definition_kind: str,
    patch_type: type,
    localns: Mapping[str, Any],
    allow_gadget_scalars: bool = False,
    type_hints: Mapping[str, Any] | None = None,
) -> tuple[Signature, Mapping[str, Any]]:
    """Resolve and validate one gadget/protocol signature without tracing it."""

    if definition_kind not in ("gadget", "protocol"):
        raise ValueError(f"unknown P2 definition kind {definition_kind!r}")
    decorator = f"@cudaq.logical.{definition_kind}"
    definition_signature = signature(provider)
    hints = {}
    input_expected = _input_expectation(definition_kind, allow_gadget_scalars)
    for name, parameter in definition_signature.parameters.items():
        if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            marker = "*" if parameter.kind is Parameter.VAR_POSITIONAL else "**"
            raise _slot_error(
                decorator,
                provider,
                f"parameter {marker}{name!s}",
                "an unsupported variadic kind",
                repr(parameter.kind.description),
                "a fixed, explicitly annotated parameter list",
            )
        if parameter.annotation is Signature.empty:
            raise _slot_error(
                decorator,
                provider,
                f"parameter {name!r}",
                "a missing annotation",
                "",
                input_expected,
            )
        annotation = _resolve_slot(
            provider,
            decorator=decorator,
            slot=f"parameter {name!r}",
            annotation=(type_hints[name] if type_hints is not None and
                        name in type_hints else parameter.annotation),
            expected=input_expected,
            localns=localns,
        )
        _validate_input(
            annotation,
            decorator=decorator,
            provider=provider,
            name=name,
            definition_kind=definition_kind,
            patch_type=patch_type,
            allow_gadget_scalars=allow_gadget_scalars,
        )
        hints[name] = annotation

    return_annotation = definition_signature.return_annotation
    if return_annotation is Signature.empty:
        raise _slot_error(
            decorator,
            provider,
            "return annotation",
            "a missing annotation",
            "",
            _OUTPUT,
        )
    resolved_return = _resolve_slot(
        provider,
        decorator=decorator,
        slot="return annotation",
        annotation=(type_hints["return"] if type_hints is not None and
                    "return" in type_hints else return_annotation),
        expected=_OUTPUT,
        localns=localns,
    )
    _validate_output(
        resolved_return,
        decorator=decorator,
        provider=provider,
        patch_type=patch_type,
    )
    hints["return"] = resolved_return
    return definition_signature, MappingProxyType(hints)
