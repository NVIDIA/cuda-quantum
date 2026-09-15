# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import replace
from cudaq.logical.programs.builder import UnplacedBuilder
from cudaq.logical.programs.definition import (
    DefinitionHandle,
    ProgramDefinition,
)
from cudaq.logical.experiments.definition import Experiment
from .build import Build, EvidenceRecord, _root_scoped_facet_names
from .context import CompilationContext
from .pipeline import Pipeline, pipelines


def _entry_device(definition):
    """Return the device captured by an objective-free entry gadget/profile."""

    device = getattr(definition, "device", None)
    if device is not None:
        return device
    experiment = getattr(definition, "experiment", None)
    device = (None if experiment is None or
              getattr(experiment, "device_provenance", None) != "entry_gadget"
              else experiment.device)
    if device is not None:
        return device
    gadget = getattr(definition, "gadget", None)
    return None if gadget is None else getattr(gadget, "device", None)


def _resolve_qec_policy(policy, values):
    """Normalize request-local P2 policy sugar against a concrete P1 schema."""

    from cudaq.logical.codes import QECBlockRequest

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
        producer="cudaq-logical-python@0.3",
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
    operating_point=None,
    module=None,
    placement=(),
    constraints=None,
    objective=None,
    noise=None,
    target=None,
    policy=None,
    parameters=None,
    _experiment=None,
    _transient=False,
) -> Build:
    from cudaq.logical.codes import (
        Code,
        CodeProfile,
        Encoding,
        EncodingEpoch,
        EncodingEpochSchema,
        EncodingHierarchy,
        EncodingProjection,
    )
    from cudaq.logical.gadgets import (
        GadgetDefinition,
        GadgetProfile,
    )
    from cudaq.logical.protocols.definition import ProtocolDefinition
    from cudaq.logical.devices.definition import Device
    from cudaq.logical.architecture.physical_definition import (
        PhysicalAction,
        PhysicalInstrument,
        PhysicalMachine,
        PhysicalDefinition,
    )
    from cudaq.logical.qec.lowering import QECLowering
    from ..targets import Target

    # An explicit lattice-surgery problem is not itself a compile root:
    # ``solve`` must first freeze its exact provider artifact. Ordinary P1 QEC
    # compilation may independently select a device-linked network compiler
    # and perform that solve through ``lower_qec``.
    from ..qec.lattice_surgery import (
        LatticeSurgeryPlan,
        LatticeSurgeryProblem,
        materialize as materialize_lattice_surgery,
    )

    if isinstance(definition, LatticeSurgeryProblem):
        raise TypeError(
            "compile() requires a solved lattice-surgery plan; call "
            "cudaq.logical.qec.lattice_surgery.solve(problem, device=device) "
            "first")

    if isinstance(definition, LatticeSurgeryPlan):
        if device is None:
            raise TypeError("lattice-surgery plan compilation requires device=")
        unsupported = {
            "operating_point": operating_point,
            "module": module,
            "placement": tuple(placement or ()),
            "constraints": constraints,
            "objective": objective,
            "noise": noise,
            "target": target,
            "policy": policy,
            "parameters": parameters,
            "experiment": _experiment,
        }
        supplied = tuple(name for name, value in unsupported.items()
                         if value not in (None, ()))
        if supplied:
            raise TypeError(
                "lattice-surgery plan compilation does not accept " +
                ", ".join(supplied))
        p2 = materialize_lattice_surgery(
            definition,
            device=device,
        )
        return compile(
            p2,
            device=device,
            pipeline=pipeline,
        )

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
        if operating_point is None:
            operating_point = _experiment.operating_point
        elif _experiment.operating_point is not None:
            raise TypeError(
                "operating_point= conflicts with the explicit Experiment")
        if not placement:
            placement = _experiment.placement
        elif _experiment.placement:
            raise TypeError("placement= conflicts with the explicit Experiment")
        if objective is None:
            objective = _experiment.objective
        if noise is None:
            noise = _experiment.noise
        if target is None:
            target = _experiment.target
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

    if operating_point is not None:
        from cudaq.logical.devices.definition import PhysicalOperatingPoint

        if not isinstance(operating_point, PhysicalOperatingPoint):
            raise TypeError(
                "operating_point= requires a PhysicalOperatingPoint")
        if not isinstance(device, Device):
            raise TypeError(
                "operating_point= requires compilation against a Device")
        device = device.with_operating_point(operating_point)

    if _experiment is None and not isinstance(definition, Target):
        _experiment = Experiment(
            root=definition,
            device=device,
            device_provenance=("entry_gadget"
                               if decorated_device is not None else None),
            operating_point=operating_point,
            # A placement callback is request-local normalization sugar.  Its
            # returned typed constraints are immediately reduced to the P1
            # witness; the callback itself must never enter a manifest.
            placement=() if callable(placement) else tuple(placement or ()),
            noise=noise,
            target=target,
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
            if any(value is not None for value in (
                    noise,
                    target,
                    policy,
            )):
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
            if any(value is not None for value in (
                    noise,
                    target,
                    policy,
                    parameters,
            )):
                raise TypeError(
                    "PBC normalization accepts only an existing synthesized "
                    "P0 Build")
            from .pbc import _to_pbc

            return _to_pbc(definition, pipeline=pipeline)

        if tuple(item.name for item in pipeline.passes) == tuple(
                item.name for item in pipelines.clifford_frame().passes):
            if device is not None:
                raise TypeError(
                    "device-free Clifford-frame normalization does not "
                    "accept device=")
            if placement or constraints is not None or objective is not None:
                raise TypeError(
                    "Clifford-frame normalization does not accept placement "
                    "options")
            if any(value is not None for value in (
                    noise,
                    target,
                    policy,
                    parameters,
            )):
                raise TypeError(
                    "Clifford-frame normalization accepts only an existing "
                    "P0 Build")
            from .frame import _absorb_clifford_frame

            return _absorb_clifford_frame(definition, pipeline=pipeline)

    if isinstance(definition, Target):
        if pipeline is not None:
            raise TypeError(
                "target manifest materialization does not take a pipeline")
        from .target_manifest import materialize_target

        return materialize_target(definition, module=module)

    # A selected P2 build asks its device for the compatible physical projector
    # and exact default P3 recipe.  This is the ordinary spelling used by
    # lattice-surgery plan composition; callers may still pass an accepted
    # explicit P3 pipeline.
    if (pipeline is None and isinstance(definition, Build) and
            definition.profile in {"p2a", "p2n"} and device is not None):
        from .physical_lower import physical_projection_pipeline

        pipeline = physical_projection_pipeline(definition, device=device)

    # Physical projection is a cross-stage compiler route. First obtain the
    # definition's natural selected P2 product, then project that immutable
    # closure against the concrete device architecture.
    if (pipeline is not None and pipeline.output_profile == "p3" and
            not isinstance(definition, (Device, PhysicalDefinition))):
        if device is None:
            raise TypeError("P2-to-P3 physical projection requires device=")
        if isinstance(definition, Build):
            source = definition
            if source.profile == "p0":
                source = compile(
                    source,
                    pipeline=pipelines.qec(),
                    device=device,
                    placement=placement,
                    constraints=constraints,
                    objective=objective,
                    policy=policy,
                    _experiment=_experiment,
                    _transient=True,
                )
            elif source.profile == "p1":
                source = compile(
                    source,
                    pipeline=pipelines.qec(),
                    device=device,
                    policy=policy,
                    _experiment=_experiment,
                    _transient=True,
                )
        elif isinstance(definition, (GadgetDefinition, GadgetProfile)):
            source = compile(
                definition,
                pipeline=pipelines.gadgets(),
                _experiment=_experiment,
                _transient=True,
            )
        elif isinstance(definition, ProtocolDefinition):
            source = compile(
                definition,
                pipeline=pipelines.protocols(),
                _experiment=_experiment,
                _transient=True,
            )
        elif isinstance(definition, ProgramDefinition):
            source = compile(
                definition,
                pipeline=pipelines.qec(),
                device=device,
                placement=placement,
                constraints=constraints,
                objective=objective,
                policy=policy,
                _experiment=_experiment,
                _transient=True,
            )
        else:
            raise TypeError(
                "physical projection expects a logical, gadget, protocol, or P2 Build"
            )
        if source.profile not in {"p2a", "p2n"}:
            raise ValueError(
                f"physical projection requires P2A/P2N input, got {source.profile}"
            )
        from .physical_lower import project_physical

        physical = project_physical(
            source,
            device=device,
            pipeline=pipeline,
            experiment=_experiment,
            _transient=_transient,
        )
        return physical

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
                producer="cudaq-logical-python@0.3",
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
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=("versioned-provider", "typed-objective",
                             "dependencies"),
            ),),
        )
    if isinstance(definition, (GadgetDefinition, GadgetProfile)):
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
                    kind=("gadget_profile_verification" if isinstance(
                        definition, GadgetProfile) else "gadget_verification"),
                    producer="cudaq-logical-python@0.3",
                    result="pass",
                    obligations=(("typed-boundary", "logical-objective",
                                  "stable-record-expressions") if isinstance(
                                      definition, GadgetProfile) else
                                 ("typed-boundary", "logical-objective")),
                ),
                linearity,
            ),
            experiment=_experiment,
            _transient=_transient,
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
                    producer="cudaq-logical-python@0.3",
                    result="pass",
                    obligations=("typed-calls", "folded-control"),
                ),
                linearity,
            ),
            experiment=_experiment,
            _transient=_transient,
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
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=(("machine-projection", "qec-space-bindings")
                             if definition.physical is None else
                             ("machine-binding", "physical-resources",
                              "topology")),
            ),),
        )
    if isinstance(definition, PhysicalAction):
        pipeline = pipeline or pipelines.device()
        if pipeline.output_profile != "p3":
            raise ValueError(
                "physical-action materialization requires a P3 pipeline")
        if device is not None:
            raise TypeError(
                "physical-action materialization does not accept device=")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p3",
            pipeline=pipeline,
            evidence=(EvidenceRecord(
                kind="physical_action_verification",
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=("positive-arity", "target-semantics"),
            ),),
        )
    if isinstance(definition, PhysicalInstrument):
        pipeline = pipeline or pipelines.device()
        if pipeline.output_profile != "p3":
            raise ValueError(
                "physical-instrument materialization requires a P3 pipeline")
        if device is not None:
            raise TypeError(
                "physical-instrument materialization does not accept device=")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p3",
            pipeline=pipeline,
            evidence=(EvidenceRecord(
                kind="physical_instrument_verification",
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=(
                    "typed-arity",
                    "record-schema",
                    "ownership-map",
                    "target-semantics",
                ),
            ),),
        )
    if isinstance(definition, PhysicalMachine):
        pipeline = pipeline or pipelines.device_stack("p3")
        if pipeline.output_profile != "p3":
            raise ValueError(
                "physical-machine materialization requires a P3 pipeline")
        if device is not None:
            raise TypeError(
                "physical-machine materialization does not accept device=")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p3",
            pipeline=pipeline,
            evidence=(EvidenceRecord(
                kind="physical_machine_verification",
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=(
                    "physical-resources",
                    "carrier-topology",
                ),
            ),),
        )
    if isinstance(definition, PhysicalDefinition):
        pipeline = pipeline or pipelines.physical()
        if pipeline.output_profile != "p3":
            raise ValueError(
                "physical graph materialization requires a P3 pipeline")
        if device is not None:
            raise TypeError(
                "@cudaq.logical.physical already binds its architecture")
        transaction = CompilationContext(module=module)
        handle = transaction.materialize(definition)
        # Linear ownership of physical states, resource payloads, and linear
        # events is verified by the real linear-use analysis, never asserted
        # unchecked.
        linearity = _verified_linearity_evidence(
            transaction.module,
            obligation="linear-resource-ownership",
            subject=f"physical graph @{handle.symbol}",
        )
        root_operation = transaction.find_symbol(handle.symbol, "phys.graph")
        return Build(
            context=transaction.context,
            module=transaction.module,
            root=handle,
            profile="p3",
            facets=_root_scoped_facet_names(
                transaction.module,
                root_operation,
            ),
            pipeline=pipeline,
            evidence=(
                EvidenceRecord(
                    kind="physical_graph_verification",
                    producer="cudaq-logical-python@0.3",
                    result="pass",
                    obligations=(
                        "stable-event-record-identity",
                        "architecture-binding",
                    ),
                ),
                linearity,
            ),
            experiment=_experiment,
        )
    pipeline = pipeline or (pipelines.placed()
                            if isinstance(definition, ProgramDefinition) and
                            definition.profile == "p1" else pipelines.logical())
    if isinstance(definition, Build):
        if definition.profile == pipeline.output_profile and device is None:
            rebind = explicit_experiment or any(value is not None for value in (
                noise,
                target,
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
                _transient=_transient,
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
                _transient=_transient,
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
                _transient=_transient,
            )
        raise ValueError(
            f"no QLX compilation route from {definition.profile!r} to "
            f"{pipeline.output_profile!r}")
    if not isinstance(definition, ProgramDefinition):
        raise TypeError(
            "cudaq.logical.compile expects a Logical definition or Build")
    if definition.profile == "p1":
        if pipeline.output_profile not in {"p1", "p2n"}:
            raise ValueError(
                "machine-bound @cudaq.logical.program definitions require a "
                "P1 or P2 pipeline")
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
            cudaq_kernel=definition.cudaq_kernel,
        )
        p0 = compile(
            portable,
            pipeline=pipelines.logical(),
            _experiment=_experiment,
            _transient=_transient,
        )
        from .place import place

        p1 = place(
            p0,
            device=definition.machine,
            placement=placement,
            constraints=constraints,
            objective=objective,
            experiment=_experiment,
            _transient=_transient,
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
            _transient=_transient,
        )
    if definition.profile != "p0":
        raise ValueError(
            f"unsupported QLX definition profile {definition.profile!r}")
    if pipeline.output_profile == "p2n":
        if device is None:
            raise TypeError("P0-to-P2 compilation requires device=")
        p0 = compile(
            definition,
            pipeline=pipelines.logical(),
            _experiment=_experiment,
            _transient=_transient,
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
            _transient=_transient,
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
            _transient=_transient,
        )
    if pipeline.output_profile == "p1":
        if device is None:
            raise TypeError("P0-to-P1 compilation requires device=")
        p0 = compile(
            definition,
            pipeline=pipelines.logical(),
            _experiment=_experiment,
            _transient=_transient,
        )
        from .place import place

        return place(
            p0,
            device=device,
            placement=placement,
            constraints=constraints,
            objective=objective,
            experiment=_experiment,
            _transient=_transient,
        )
    if pipeline.output_profile != "p0":
        raise ValueError(f"no QLX compilation route from 'p0' to "
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
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=("p0-signature", "p0-machine-free"),
            ),
            linearity,
        ),
        value_groups=transaction.value_groups_of(definition),
        source_modules=(getattr(definition, "__module__", None) or "__main__",),
        experiment=_experiment,
        _transient=_transient,
    )


def materialize(definition, module=None):
    return compile(definition, module=module)
