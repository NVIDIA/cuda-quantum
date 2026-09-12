# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from importlib import import_module

from .build import Build, EvidenceRecord, SynthesisSummary
from .link_check import (
    LinkageError,
    check_linkage,
    verify_linked,
)
from .compile import compile, materialize
from .synthesize import synthesize
from .pbc import to_pbc
from .frame import absorb_clifford_frame
from .context import CompilationContext
from .experiment_bundle import ExperimentBundle, compile_many
from .pipeline import PassSpec, Pipeline, passes, pipelines
from .place import place
from .quake import convert_quake_to_p0, import_cudaq, import_quake
from .physical_lower import (
    PhysicalProjectionBuilder,
    PhysicalProjectionEmission,
    PhysicalProjector,
    physical_projection_pipeline,
    project_physical,
)
from .qec_lower import lower_qec
from .schedule import (
    PhysicalSchedule,
    ScheduleEntry,
    SchedulingStrategy,
    schedule,
    scheduling,
)
from .factory import factory_model
from .component_models import spacetime_plan_model, transport_model
from .ticks import Tick, ticks
from .idle import add_idle, remove_idle
from .projection import (
    CompiledInterfaceManifest,
    ProjectedMeasurement,
    ProjectedPort,
)
from .topology_view import (
    CarrierGraphView,
    DeviceStackGraphView,
    GraphEdge,
    GraphNode,
    MachineGraphView,
    PatchGraphView,
    PatchTopologyView,
)
from .gate_sets import GateSet, clifford_t
from cudaq.logical.qec.lowering import (
    ActionSiteHandle,
    GeneratedQECArtifact,
    QECCompiler,
    QECCompilerContext,
    QECLowering,
    QECNetworkCompiler,
    QECNetworkContext,
    qec_lowering,
)
from cudaq.logical.experiments.definition import Experiment
from cudaq.logical.compiler.authoring import (
    ActionBuilder,
    CodeBuilder,
    GadgetBuilder,
    GadgetProfileBuilder,
    PhysicalBuilder,
    PlacedBuilder,
    ProtocolBuilder,
    UnplacedBuilder,
)


def lower(
        build,
        *,
        device=None,
        pipeline=None,
        placement=(),
        constraints=None,
        objective=None,
):
    """Continue a build through an explicit compiler pipeline.

    This is the canonical compiler-owned spelling of the historical callable
    ``cudaq.logical.lower`` module. Delegation retains one implementation of default
    pipeline selection and validation.
    """

    lowering = import_module("cudaq.logical.lower")
    return lowering(
        build,
        device=device,
        pipeline=pipeline,
        placement=placement,
        constraints=constraints,
        objective=objective,
    )


__all__ = [
    "Build",
    "EvidenceRecord",
    "SynthesisSummary",
    "CompilationContext",
    "ExperimentBundle",
    "compile_many",
    "PassSpec",
    "Pipeline",
    "compile",
    "synthesize",
    "to_pbc",
    "absorb_clifford_frame",
    "materialize",
    "lower",
    "passes",
    "pipelines",
    "place",
    "import_quake",
    "import_cudaq",
    "convert_quake_to_p0",
    "PhysicalProjector",
    "PhysicalProjectionBuilder",
    "PhysicalProjectionEmission",
    "physical_projection_pipeline",
    "project_physical",
    "lower_qec",
    "PhysicalSchedule",
    "ScheduleEntry",
    "SchedulingStrategy",
    "Tick",
    "ticks",
    "add_idle",
    "remove_idle",
    "schedule",
    "factory_model",
    "spacetime_plan_model",
    "transport_model",
    "scheduling",
    "CompiledInterfaceManifest",
    "ProjectedMeasurement",
    "ProjectedPort",
    "GraphNode",
    "GraphEdge",
    "MachineGraphView",
    "PatchTopologyView",
    "DeviceStackGraphView",
    "PatchGraphView",
    "CarrierGraphView",
    "GateSet",
    "clifford_t",
    "ActionBuilder",
    "CodeBuilder",
    "GadgetBuilder",
    "GadgetProfileBuilder",
    "PhysicalBuilder",
    "PlacedBuilder",
    "UnplacedBuilder",
    "ProtocolBuilder",
    "Experiment",
    "GeneratedQECArtifact",
    "QECCompiler",
    "QECNetworkCompiler",
    "QECNetworkContext",
    "QECLowering",
    "ActionSiteHandle",
    "QECCompilerContext",
    "qec_lowering",
]
