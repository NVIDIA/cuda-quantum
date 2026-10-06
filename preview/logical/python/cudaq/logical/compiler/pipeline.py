# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable

from cudaq.logical._core.immutable import freeze_value

if TYPE_CHECKING:
    from cudaq.logical.stages import Stage


@dataclass(frozen=True, slots=True)
class PassSpec:
    name: str
    options: tuple[tuple[str, Any], ...] = ()
    requires_facets: tuple[str, ...] = ()
    provides_facets: tuple[str, ...] = ()
    preserves_facets: tuple[str, ...] = ()
    invalidates_facets: tuple[str, ...] = ()
    recomputes_facets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "options",
            tuple((key, freeze_value(value)) for key, value in self.options),
        )
        for name in (
                "requires_facets",
                "provides_facets",
                "preserves_facets",
                "invalidates_facets",
                "recomputes_facets",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class Pipeline:
    """An immutable ordered QLX compiler pipeline."""

    passes: tuple[PassSpec, ...]
    output_profile: str | Stage

    def __post_init__(self) -> None:
        object.__setattr__(self, "passes", tuple(self.passes))

    @property
    def output_stage(self):
        from cudaq.logical.stages import stage_and_facets

        return stage_and_facets(self.output_profile)[0]

    @property
    def provided_facets(self):
        _, legacy = self._output_product()
        return self.apply_facets(legacy)

    def _output_product(self):
        from cudaq.logical.stages import stage_and_facets

        return stage_and_facets(self.output_profile)

    def apply_facets(self, initial=()):
        """Apply declared facet effects without imposing a facet order.

        Facets survive a pass unless it explicitly invalidates them. A
        recomputed facet is first invalidated and then re-established.
        """
        from cudaq.logical.stages import normalize_facets

        current = list(normalize_facets(initial))
        for item in self.passes:
            removed = normalize_facets(
                (*item.invalidates_facets, *item.recomputes_facets))
            current = [facet for facet in current if facet not in removed]
            added = normalize_facets(
                (*item.provides_facets, *item.recomputes_facets))
            for facet in added:
                if facet not in current:
                    current.append(facet)
        return tuple(current)

    def insert_after(self, pass_name: str, new_pass: PassSpec) -> "Pipeline":
        items = list(self.passes)
        for index, item in enumerate(items):
            if item.name == pass_name:
                items.insert(index + 1, new_pass)
                return replace(self, passes=tuple(items))
        raise KeyError(f"pipeline contains no pass named {pass_name!r}")

    def configure(self, pass_name: str, **options) -> "Pipeline":
        items = list(self.passes)
        for index, item in enumerate(items):
            if item.name == pass_name:
                merged = dict(item.options)
                merged.update(options)
                items[index] = replace(item, options=tuple(merged.items()))
                return replace(self, passes=tuple(items))
        raise KeyError(f"pipeline contains no pass named {pass_name!r}")

    def replace(self, pass_name: str, new_pass: PassSpec) -> "Pipeline":
        if not isinstance(new_pass, PassSpec):
            raise TypeError("pipeline replacement must be a PassSpec")
        items = list(self.passes)
        for index, item in enumerate(items):
            if item.name == pass_name:
                items[index] = new_pass
                return replace(self, passes=tuple(items))
        raise KeyError(f"pipeline contains no pass named {pass_name!r}")

    def append(self, new_pass: PassSpec) -> "Pipeline":
        if not isinstance(new_pass, PassSpec):
            raise TypeError("pipeline append requires a PassSpec")
        return replace(self, passes=(*self.passes, new_pass))

    def remove(self, pass_name: str) -> "Pipeline":
        items = tuple(item for item in self.passes if item.name != pass_name)
        if len(items) == len(self.passes):
            raise KeyError(f"pipeline contains no pass named {pass_name!r}")
        return replace(self, passes=items)


class _Pipelines:

    def logical(self) -> Pipeline:
        return Pipeline(
            passes=(
                PassSpec("qlx-normalize-actions"),
                PassSpec("qlx-infer-requirements"),
                PassSpec("qlx-verify-p0"),
            ),
            output_profile="p0",
        )

    def clifford_t(self, *, precision: float = 1.0e-10) -> Pipeline:
        """Device-free P0 legalization to positive H/S/T/CX."""

        from ..compiler.gate_sets import clifford_t

        return clifford_t.pipeline(precision=precision)

    def pbc(self) -> Pipeline:
        """Device-free P0 normalization of Clifford+T into PBC form."""

        return Pipeline(
            passes=(
                PassSpec("qlx-to-pbc"),
                PassSpec("qlx-verify-pbc"),
                PassSpec("qlx-verify-p0"),
            ),
            output_profile="p0",
        )

    def clifford_frame(self) -> Pipeline:
        """Absorb exact Cliffords while preserving arbitrary rotations."""

        return Pipeline(
            passes=(
                PassSpec("qlx-absorb-clifford-frame"),
                PassSpec("qlx-verify-clifford-frame"),
                PassSpec("qlx-verify-p0"),
            ),
            output_profile="p0",
        )

    def placed(self) -> Pipeline:
        return Pipeline(
            passes=(
                PassSpec("qlx-verify-p0"),
                PassSpec("qlx-place"),
                PassSpec("qlx-to-lvm"),
                PassSpec("lvm-verify-p1"),
            ),
            output_profile="p1",
        )

    def qec_definitions(self) -> Pipeline:
        return Pipeline(
            passes=(PassSpec("fabric-verify-p2s",
                             provides_facets=("qec_spec",)),),
            output_profile="p2s",
        )

    def gadgets(self) -> Pipeline:
        return Pipeline(
            passes=(
                PassSpec("fabric-materialize-record-schemas"),
                PassSpec(
                    "fabric-verify-p2a",
                    requires_facets=("qec_spec",),
                    provides_facets=("qec_realization",),
                ),
            ),
            output_profile="p2a",
        )

    def protocols(self) -> Pipeline:
        return Pipeline(
            passes=(
                PassSpec("fabric-link-calls"),
                PassSpec(
                    "fabric-verify-p2n",
                    provides_facets=("protocol_network",),
                ),
            ),
            output_profile="p2n",
        )

    def qec(self) -> Pipeline:
        return Pipeline(
            passes=(
                PassSpec("lvm-select"),
                PassSpec("lvm-apply-qec-lowerings"),
                PassSpec("lvm-to-fabric"),
                PassSpec("fabric-materialize-default-encodings"),
                PassSpec(
                    "fabric-verify-generated-protocols",
                    provides_facets=("protocol_network",),
                ),
            ),
            output_profile="p2n",
        )

    def device(self) -> Pipeline:
        return Pipeline(passes=(PassSpec("phys-verify-architecture"),),
                        output_profile="p3")

    def device_stack(self, profile: str) -> Pipeline:
        """Verification recipe for one immutable static device prefix."""

        profile = str(profile)
        passes = {
            "p1": (PassSpec("lvm-verify-p1"),),
            "p2": (PassSpec("fabric-verify-machine"),),
            "p3": (PassSpec("phys-verify-machine"),),
        }
        try:
            recipe = passes[profile]
        except KeyError as exc:
            raise ValueError(
                "device stack profile must be p1, p2, or p3") from exc
        return Pipeline(passes=recipe, output_profile=profile)

    def physical(self) -> Pipeline:
        return Pipeline(
            passes=(
                PassSpec(
                    "fabric-derive-patch-graph",
                    provides_facets=("patch_graph",),
                ),
                PassSpec(
                    "fabric-map-patches",
                    requires_facets=("patch_graph",),
                ),
                PassSpec("fabric-to-phys"),
                PassSpec(
                    "phys-route",
                    provides_facets=("carrier_mapping", "physical_routing"),
                ),
                PassSpec(
                    "phys-legalize-native-actions",
                    provides_facets=("native_legalization",),
                ),
                PassSpec(
                    "phys-verify-p3",
                    invalidates_facets=("physical_schedule",),
                ),
            ),
            output_profile="p3",
        )


class _Passes:

    def recognize_actions(self, *actions) -> PassSpec:
        return PassSpec(
            "qlx-recognize-actions",
            (("actions", tuple(action.name for action in actions)),),
        )


pipelines = _Pipelines()
passes = _Passes()
