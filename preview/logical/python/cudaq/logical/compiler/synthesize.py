# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Device-free P0 logical gate-set legalization."""

from __future__ import annotations

from ..compiler.gate_sets import GateSet
from ..programs.definition import ProgramDefinition
from .build import Build, EvidenceRecord


def _pass_pipeline_string(pass_spec, gate_set: GateSet) -> str:
    options = dict(pass_spec.options)
    if pass_spec.name == "qlx-synthesize-rotations":
        return f"qlx-synthesize-rotations{{precision={options['precision']!r}}}"
    if pass_spec.name == "qlx-verify-clifford-t":
        return pass_spec.name
    raise ValueError(f"gate set {gate_set.name!r} requires unsupported "
                     f"legalization pass {pass_spec.name!r}")


def _execute_pipeline(source: Build, gate_set: GateSet, pipeline) -> Build:
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = source._fresh_module()
    joined_pipeline = ",".join(
        _pass_pipeline_string(pass_spec, gate_set)
        for pass_spec in pipeline.passes)
    runtime.run_pass(module, joined_pipeline)

    precision = dict(pipeline.passes[0].options)["precision"]
    return Build(
        context=module.context,
        module=module,
        root=source.root,
        profile="p0",
        facets=source.facets,
        pipeline=pipeline,
        evidence=(
            *source.evidence,
            EvidenceRecord(
                kind="logical_gate_set_legalization",
                producer="cudaq-synth/gridsynth",
                result="pass",
                obligations=(
                    f"gate-set:{gate_set.name}",
                    "positive-generators:h,s,t,cx",
                    "static-pauli-rotations",
                    "projective-operator-norm",
                ),
                assumptions=(
                    f"default-precision:{precision:.17g}",
                    "authored-site-precision-overrides-default",
                    "global-phase-ignored",
                ),
            ),
        ),
        value_groups={
            name: len(group) for name, group in source.values._groups.items()
        },
        experiment=source.experiment,
        source_modules=source.source_modules,
    )


def _synthesize(
    definition,
    *,
    gate_set: GateSet,
    precision: float,
    parameters=None,
    pipeline=None,
    experiment=None,
) -> Build:
    if not isinstance(gate_set, GateSet):
        raise TypeError("synthesize gate_set= must be a cudaq.logical.GateSet")
    pipeline = pipeline or gate_set.pipeline(precision=precision)

    if isinstance(definition, Build):
        if definition.profile != "p0":
            raise ValueError(
                "cudaq.logical.synthesize requires a qlx.program or P0 Build, "
                f"got {definition.profile!r}")
        if parameters:
            raise TypeError(
                "parameters= can specialize a qlx.program, not an existing "
                "Build; synthesize the ProgramDefinition directly")
        source = definition
    else:
        if not isinstance(definition, ProgramDefinition):
            raise TypeError(
                "cudaq.logical.synthesize expects a @cudaq.logical.program definition or P0 Build"
            )
        from .compile import compile
        from .pipeline import pipelines

        source = compile(
            definition,
            pipeline=pipelines.logical(),
            parameters=parameters,
            _experiment=experiment,
        )
    return _execute_pipeline(source, gate_set, pipeline)


def synthesize(
    definition,
    *,
    gate_set: GateSet,
    precision: float = 1.0e-10,
    parameters=None,
) -> Build:
    """Legalize a logical program to ``gate_set`` without selecting a device.

    ``precision`` is the default projective operator-norm bound for each
    off-lattice rotation. A rotation's explicitly authored ``precision=``
    takes precedence. Runtime-dependent angles must be specialized through
    ``parameters=``.
    """

    if not isinstance(gate_set, GateSet):
        raise TypeError("synthesize gate_set= must be a cudaq.logical.GateSet")
    pipeline = gate_set.pipeline(precision=precision)
    return _synthesize(
        definition,
        gate_set=gate_set,
        precision=precision,
        parameters=parameters,
        pipeline=pipeline,
    )


__all__ = ["synthesize"]
