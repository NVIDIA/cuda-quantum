# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..programs.builder import UnplacedBuilder
from ..programs.definition import (
    DefinitionHandle,
    ProgramDefinition,
)
from ..experiments.definition import Experiment
from .build import Build, EvidenceRecord
from .context import CompilationContext
from .pipeline import Pipeline, pipelines


def _entry_device(definition):
    """Return the device captured by an objective-free entry gadget."""

    device = getattr(definition, "device", None)
    if device is not None:
        return device
    experiment = getattr(definition, "experiment", None)
    device = (None if experiment is None or
              getattr(experiment, "device_provenance", None) != "entry_gadget"
              else experiment.device)
    if device is not None:
        return device
    return None


def _resolve_qec_policy(policy, values):
    """Normalize request-local P2 policy sugar against a concrete P1 schema."""

    from ..codes import QECBlockRequest

    value = policy(values) if callable(policy) else policy
    if isinstance(value, QECBlockRequest):
        return {"qec_blocks": (value,)}
    if isinstance(value, (tuple, list)) and all(
            isinstance(item, QECBlockRequest) for item in value):
        return {"qec_blocks": tuple(value)}
    return value


def _verified_linearity_evidence(module, *, obligation,
                                 subject) -> EvidenceRecord:
    """Run the real linear-ownership analysis and mint honest evidence.

    Raises ``LinearityError`` (with per-value diagnostics) when any body in
    the module violates single ownership, so a ``pass`` record is only ever
    emitted for IR the analysis actually accepted.
    """
    from .linearity import verify_linearity

    report = verify_linearity(module, subject=subject)
    return EvidenceRecord(
        kind="linearity_verification",
        producer="qlx-python@0.3",
        result=report.result,
        obligations=(obligation,),
        assumptions=(
            "checker:qlx-linear-use/v1",
            f"checked-bodies:{len(report.checked_bodies)}",
            f"linear-values:{report.linear_values}",
        ),
    )


def compile(
    definition,
    *,
    pipeline: Pipeline | None = None,
    device=None,
    module=None,
    placement=(),
    constraints=None,
    objective=None,
    policy=None,
    parameters=None,
    _experiment=None,
) -> Build:
    from ..codes import (
        Code,
        CodeProfile,
        Encoding,
        EncodingEpoch,
        EncodingEpochSchema,
        EncodingHierarchy,
        EncodingProjection,
    )
    from ..gadgets import GadgetDefinition
    from ..protocols.definition import ProtocolDefinition
    from ..devices.definition import Device
    from ..qec.lowering import QECLowering

    explicit_experiment = isinstance(definition, Experiment)
    if explicit_experiment:
        if _experiment is not None:
            raise TypeError(
                "nested cudaq.logical.Experiment values are not supported")
        _experiment = definition
        definition = _experiment.root
        if device is None:
            device = _experiment.device
        elif (_experiment.device is not None and
              device is not _experiment.device and
              not (isinstance(_experiment.device, Device) and
                   isinstance(device, Device) and
                   _experiment.device._has_same_static_stack(device))):
            raise TypeError("device= conflicts with the explicit Experiment")
        if not placement:
            placement = _experiment.placement
        elif _experiment.placement:
            raise TypeError("placement= conflicts with the explicit Experiment")
        if objective is None:
            objective = _experiment.objective
        if policy is None:
            policy = _experiment.policy
        if parameters is None:
            parameters = _experiment.parameters

    decorated_device = _entry_device(definition)
    if decorated_device is not None:
        if device is None:
            device = decorated_device
        elif (device is not decorated_device and
              not (isinstance(decorated_device, Device) and
                   isinstance(device, Device) and
                   decorated_device._has_same_static_stack(device))):
            raise TypeError(
                "device= conflicts with the top-level @cudaq.logical.gadget device"
            )
        if (_experiment is not None and _experiment.device_provenance is None):
            _experiment = replace(
                _experiment,
                device=device,
                device_provenance="entry_gadget",
            )

    # Compile-time program specialization is explicit and signature based.
    # Request parameters that name runtime ABI arguments are bound before
    # tracing; unrelated experiment parameters remain request metadata.
    if isinstance(definition, ProgramDefinition) and parameters:
        if not hasattr(parameters, "items"):
            raise TypeError("compile parameters= must be a mapping")
        bindings = {
            name: value
            for name, value in parameters.items()
            if name in definition.signature.parameters
        }
        if bindings:
            definition = definition.specialize(**bindings)

    if _experiment is None:
        _experiment = Experiment(
            root=definition,
            device=device,
            device_provenance=("entry_gadget"
                               if decorated_device is not None else None),
            # A placement callback is request-local normalization sugar.  Its
            # returned typed constraints are immediately reduced to the P1
            # witness; the callback itself must never enter a manifest.
            placement=() if callable(placement) else tuple(placement or ()),
            policy=None if callable(policy) else policy,
            parameters=parameters,
            objective=objective,
        )

    if pipeline is not None:
        from ..compiler.gate_sets import _match_pipeline

        gate_set_match = _match_pipeline(pipeline)
        if gate_set_match is not None:
            if device is not None:
                raise TypeError(
                    "device-free logical synthesis does not accept device=")
            if placement or constraints is not None or objective is not None:
                raise TypeError(
                    "logical synthesis does not accept placement options")
            if policy is not None:
                raise TypeError("logical synthesis does not accept downstream "
                                "analysis, target, or policy options")
            gate_set, precision = gate_set_match
            from .synthesize import _synthesize

            return _synthesize(
                definition,
                gate_set=gate_set,
                precision=precision,
                parameters=parameters,
                pipeline=pipeline,
                experiment=_experiment,
            )

        if tuple(item.name for item in pipeline.passes) == tuple(
                item.name for item in pipelines.pbc().passes):
            if device is not None:
                raise TypeError(
                    "device-free PBC normalization does not accept device=")
            if placement or constraints is not None or objective is not None:
                raise TypeError(
                    "PBC normalization does not accept placement options")
            if any(value is not None for value in (policy, parameters)):
                raise TypeError(
                    "PBC normalization accepts only an existing synthesized "
                    "P0 Build")
            from .pbc import _to_pbc

            return _to_pbc(definition, pipeline=pipeline)

    if isinstance(
            definition,
        (Code, CodeProfile, Encoding, EncodingEpoch, EncodingEpochSchema,
         EncodingHierarchy, EncodingProjection),
    ):
        pipeline = pipeline or pipelines.qec_definitions()
        if pipeline.output_profile != "p2s":
            raise ValueError(
                "QEC definition materialization requires a P2S pipeline")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p2s",
            pipeline=pipeline,
            evidence=(EvidenceRecord(
                kind="code_algebra_verification",
                producer="qlx-python@0.3",
                result="pass",
                obligations=("code-shape", "default-profile",
                             "default-encoding"),
            ),),
        )
    if isinstance(definition, QECLowering):
        pipeline = pipeline or pipelines.qec_definitions()
        if pipeline.output_profile != "p2s":
            raise ValueError(
                "QEC-lowering manifest materialization requires a P2S "
                "verification recipe")
        # The pass verifies P2S dependencies, but the selected manifest root
        # itself is stage-neutral. Record the actual root profile in the
        # replay recipe so the v2 classification is self-consistent.
        pipeline = replace(pipeline, output_profile="common")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="common",
            pipeline=pipeline,
            evidence=(EvidenceRecord(
                kind="qec_lowering_manifest_verification",
                producer="qlx-python@0.3",
                result="pass",
                obligations=("versioned-provider", "typed-objective",
                             "dependencies"),
            ),),
        )
    if isinstance(definition, GadgetDefinition):
        pipeline = pipeline or pipelines.gadgets()
        if pipeline.output_profile != "p2a":
            raise ValueError("gadget materialization requires a P2A pipeline")
        if device is not None and _entry_device(definition) is None:
            raise TypeError(
                "a reusable gadget definition does not accept device=")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        # The linear-patch-ownership obligation is discharged by the real
        # linear-use analysis over the materialized gadget bodies, never
        # asserted unchecked.
        linearity = _verified_linearity_evidence(
            transaction.module,
            obligation="linear-patch-ownership",
            subject=f"gadget @{handle.symbol}",
        )
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p2a",
            pipeline=pipeline,
            evidence=(
                EvidenceRecord(
                    kind="gadget_verification",
                    producer="qlx-python@0.3",
                    result="pass",
                    obligations=("typed-boundary", "logical-objective"),
                ),
                linearity,
            ),
            experiment=_experiment,
        )
    if isinstance(definition, ProtocolDefinition):
        pipeline = pipeline or pipelines.protocols()
        if pipeline.output_profile != "p2n":
            raise ValueError("protocol materialization requires a P2N pipeline")
        if device is not None:
            raise TypeError(
                "a reusable protocol definition does not accept device=")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        # Linear ownership across the folded protocol graph is verified by
        # the real linear-use analysis, never asserted unchecked.
        linearity = _verified_linearity_evidence(
            transaction.module,
            obligation="linear-ownership",
            subject=f"protocol @{handle.symbol}",
        )
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p2n",
            pipeline=pipeline,
            evidence=(
                EvidenceRecord(
                    kind="protocol_verification",
                    producer="qlx-python@0.3",
                    result="pass",
                    obligations=("typed-calls", "folded-control"),
                ),
                linearity,
            ),
            experiment=_experiment,
        )
    if isinstance(definition, Device):
        expected_profile = definition.layers[-1].value
        pipeline = pipeline or pipelines.device_stack(expected_profile)
        if pipeline.output_profile != expected_profile:
            raise ValueError(
                f"device materialization requires a {expected_profile.upper()} pipeline"
            )
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile=expected_profile,
            pipeline=pipeline,
            evidence=(EvidenceRecord(
                kind="device_binding_verification",
                producer="qlx-python@0.3",
                result="pass",
                obligations=("machine-projection", "qec-space-bindings"),
            ),),
        )
    pipeline = pipeline or (pipelines.placed()
                            if isinstance(definition, ProgramDefinition) and
                            definition.profile == "p1" else pipelines.logical())
    if isinstance(definition, Build):
        if definition.profile == pipeline.output_profile and device is None:
            rebind = explicit_experiment or any(value is not None for value in (
                policy,
                parameters,
                objective,
            )) or bool(placement) or constraints is not None
            if not rebind:
                return definition
            replay = definition.module
            replay_context = replay.context
            return Build(
                context=replay_context,
                module=replay,
                root=definition.root,
                profile=definition.profile,
                facets=definition.facets,
                pipeline=pipeline,
                evidence=definition.evidence,
                value_groups={
                    name: len(group)
                    for name, group in definition.values._groups.items()
                },
                placement=definition.placement,
                qec_selection=definition.qec_selection,
                experiment=_experiment,
                source_modules=definition.source_modules,
            )
        if definition.profile == "p0" and pipeline.output_profile in {
                "p1", "p2n"
        }:
            if device is None:
                raise TypeError("P0 placement requires device=")
            from .place import place

            p1 = place(
                definition,
                device=device,
                placement=placement,
                constraints=constraints,
                objective=objective,
                experiment=_experiment,
            )
            if pipeline.output_profile == "p1":
                return p1
            from .qec_lower import lower_qec

            qec_policy = _resolve_qec_policy(policy, p1.values)
            if callable(policy):
                _experiment = replace(_experiment, policy=qec_policy)

            return lower_qec(
                p1,
                device=device,
                pipeline=pipeline,
                policy=qec_policy,
                experiment=_experiment,
            )
        if definition.profile == "p1" and pipeline.output_profile == "p2n":
            if device is None:
                raise TypeError("P1-to-P2 compilation requires device=")
            from .qec_lower import lower_qec

            qec_policy = _resolve_qec_policy(policy, definition.values)
            if callable(policy):
                _experiment = replace(_experiment, policy=qec_policy)

            return lower_qec(
                definition,
                device=device,
                pipeline=pipeline,
                policy=qec_policy,
                experiment=_experiment,
            )
        raise ValueError(
            f"no CUDA-Q Logical compilation route from {definition.profile!r} to "
            f"{pipeline.output_profile!r}")
    if not isinstance(definition, ProgramDefinition):
        raise TypeError(
            "cudaq.logical.compile expects a CUDA-Q Logical definition or Build"
        )
    if definition.profile == "p1":
        if pipeline.output_profile not in {"p1", "p2n"}:
            raise ValueError(
                "machine-bound @cudaq.logical.program definitions require a P1 or P2 pipeline"
            )
        # The same provider is traced once into machine-free intent and then
        # refined by the placement pass. Explicit P1-only helpers are added in
        # the next builder slice; ordinary logical source already shares this
        # exact path.
        portable = ProgramDefinition(
            definition.provider,
            machine=None,
            selection=definition.selection,
            kind=definition.kind,
            objective_kind=definition.objective_kind,
            name=definition.name,
            metadata=definition.metadata,
            type_hints=definition.type_hints,
            specialization=definition.specialization,
            base=definition.base,
        )
        p0 = compile(
            portable,
            pipeline=pipelines.logical(),
            _experiment=_experiment,
        )
        from .place import place

        p1 = place(
            p0,
            device=definition.machine,
            placement=placement,
            constraints=constraints,
            objective=objective,
            experiment=_experiment,
        )
        if pipeline.output_profile == "p1":
            return p1
        if device is None:
            raise TypeError("P1-to-P2 compilation requires a concrete device=")
        from .qec_lower import lower_qec

        qec_policy = _resolve_qec_policy(policy, p1.values)
        if callable(policy):
            _experiment = replace(_experiment, policy=qec_policy)

        return lower_qec(
            p1,
            device=device,
            pipeline=pipeline,
            policy=qec_policy,
            experiment=_experiment,
        )
    if definition.profile != "p0":
        raise ValueError(
            f"unsupported CUDA-Q Logical definition profile {definition.profile!r}"
        )
    if pipeline.output_profile == "p2n":
        if device is None:
            raise TypeError("P0-to-P2 compilation requires device=")
        p0 = compile(
            definition,
            pipeline=pipelines.logical(),
            _experiment=_experiment,
        )
        from .place import place
        from .qec_lower import lower_qec

        p1 = place(
            p0,
            device=device,
            placement=placement,
            constraints=constraints,
            objective=objective,
            experiment=_experiment,
        )
        qec_policy = _resolve_qec_policy(policy, p1.values)
        if callable(policy):
            _experiment = replace(_experiment, policy=qec_policy)
        return lower_qec(
            p1,
            device=device,
            pipeline=pipeline,
            policy=qec_policy,
            experiment=_experiment,
        )
    if pipeline.output_profile == "p1":
        if device is None:
            raise TypeError("P0-to-P1 compilation requires device=")
        p0 = compile(
            definition,
            pipeline=pipelines.logical(),
            _experiment=_experiment,
        )
        from .place import place

        return place(
            p0,
            device=device,
            placement=placement,
            constraints=constraints,
            objective=objective,
            experiment=_experiment,
        )
    if pipeline.output_profile != "p0":
        raise ValueError(f"no CUDA-Q Logical compilation route from 'p0' to "
                         f"{pipeline.output_profile!r}")
    if placement or constraints is not None or objective is not None:
        raise TypeError("placement options require a P1-or-later pipeline")
    if device is not None:
        raise TypeError("a logical-only pipeline does not accept device=")
    transaction = CompilationContext(module=module)
    handle = transaction.materialize(definition)
    # Every canonical P0 program body is checked for linear single ownership
    # of its logical qubits, resources, and linear events; the evidence below
    # records the actual analysis result.
    linearity = _verified_linearity_evidence(
        transaction.module,
        obligation="linear-ownership",
        subject=f"program @{handle.symbol}",
    )
    return Build(
        context=transaction.context,
        module=transaction.module,
        root=handle,
        profile="p0",
        pipeline=pipeline,
        evidence=(
            EvidenceRecord(
                kind="profile_verification",
                producer="qlx-python@0.3",
                result="pass",
                obligations=("p0-signature", "p0-machine-free"),
            ),
            linearity,
        ),
        value_groups=transaction.value_groups_of(definition),
        source_modules=(definition.__module__,),
        experiment=_experiment,
    )


def materialize(definition, module=None):
    return compile(definition, module=module)
