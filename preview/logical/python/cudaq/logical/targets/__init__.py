# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

from cudaq._experimental import CompileTarget, CustomTarget

from ..lower import (
    LoweringSpec,
    emit_mlir,
    lower,
)


@dataclass(frozen=True, slots=True)
class EmitPolicy:
    pass


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    pass


_POLICY_CAPABILITY = {
    EmitPolicy: "emit_text",
    ArtifactPolicy: "emit_artifact",
}


def _finalizer_identity(finalize) -> str:
    """Return a stable public identity for one target finalizer.

    Built-in and custom finalizers retain their actual canonical module
    identity.  The optional explicit identity remains authoritative for
    versioned public providers.
    """

    explicit = getattr(finalize, "__logical_finalizer__", None)
    if explicit is not None:
        return explicit
    return f"python:{finalize.__module__}:{finalize.__qualname__}"


class UnavailableTargetError(RuntimeError):
    """A serialized target names no resolvable versioned target provider."""


class Backend:
    """One stack layer that compiles a build before delegating downstream."""

    def __init__(self, spec: LoweringSpec, *, next_backend=None):
        if not isinstance(spec, LoweringSpec):
            raise TypeError("Backend spec must be a LoweringSpec")
        if next_backend is not None and not isinstance(next_backend, Backend):
            raise TypeError("Backend next_backend must be a Backend or None")
        if next_backend is None and spec.finalize is None:
            raise ValueError("a leaf Backend requires a LoweringSpec finalizer")
        self.spec, self.next_backend = spec, next_backend

    def compile(self, build, **_options):
        return build

    def _launch(self, build, operation, **kwargs):
        prepared = self.compile(build, **kwargs)
        if self.next_backend is not None:
            return getattr(self.next_backend, operation)(prepared, **kwargs)
        module, context = lower(self.spec, prepared)
        return self.spec.finalize(module, context, **kwargs)

    def estimate(self, build, args=(), *, tier=None, **estimate_options):
        """Return this backend stack's estimates as CUDA-Q annotations."""

        return _cudaq_estimate_result(
            self._estimate(build, args, tier=tier, **estimate_options))

    def _estimate(self, build, args=(), *, tier=None, **estimate_options):
        """Estimate each available stage, or lower only to one requested tier."""

        if tier is None:
            own = _stage_estimate(build, **estimate_options)
            prepared = self.compile(build)
            if self.next_backend is None:
                return own
            downstream = self.next_backend._estimate(prepared, args,
                                                     **estimate_options)
            return _merge_estimates(own, downstream)

        tier = _estimate_tier(tier)
        own = _stage_estimate(build, tier=tier, **estimate_options)
        if own:
            return own
        if self.next_backend is None:
            raise ValueError(
                f"Tier.{tier.name} is unavailable from a backend accepting "
                f"{getattr(build, 'profile', type(build).__name__)}")
        return self.next_backend._estimate(self.compile(build),
                                           args,
                                           tier=tier,
                                           **estimate_options)

    def emit(self, build, **kwargs):
        return self._launch(build, "emit", **kwargs)

    def emit_artifact(self, build, **kwargs):
        return self._launch(build, "emit_artifact", **kwargs)

    def _launch_policy(self, build, policy, args, **kwargs):
        kwargs.setdefault("arguments", args)
        if self.next_backend is None:
            return self._launch(build, policy, **kwargs)
        return getattr(self.next_backend, policy)(build, **kwargs)

    def sample(self, build, args=(), **kwargs):
        return self._launch_policy(build, "sample", args, **kwargs)

    def observe(self, build, args=(), **kwargs):
        return self._launch_policy(build, "observe", args, **kwargs)

    def dem_from_kernel(self, build, args=(), **kwargs):
        return self._launch_policy(build, "dem_from_kernel", args, **kwargs)

    def stack_metadata(self):
        """Return display metadata owned by this backend layer."""

        return ()


class TerminalBackend(Backend):
    """Terminal stack layer selecting one finalizer by target capability."""

    def __init__(self, specs=None):
        specs = dict(specs) if specs is not None else {}
        if any(not isinstance(spec, LoweringSpec) for spec in specs.values()):
            raise TypeError("TerminalBackend specs must be LoweringSpec values")
        if any(spec.finalize is None for spec in specs.values()):
            raise ValueError("a TerminalBackend spec requires a finalizer")
        self.specs, self.next_backend = MappingProxyType(specs), None

    def _launch(self, build, operation, *, capability=None, **kwargs):
        capability = capability or operation
        try:
            spec = self.specs[capability]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__name__} has no {capability!r} capability"
            ) from exc
        module, context = lower(spec, build)
        return spec.finalize(module, context, capability=capability, **kwargs)

    @staticmethod
    def _unsupported_policy(policy):
        raise RuntimeError("cudaq.logical is in preview, and launch policy "
                           f"{policy!r} is not yet supported")

    def sample(self, _build, _args=(), **_kwargs):
        self._unsupported_policy("sample")

    def observe(self, _build, _args=(), **_kwargs):
        self._unsupported_policy("observe")

    def dem_from_kernel(self, _build, _args=(), **_kwargs):
        self._unsupported_policy("dem")

    def stack_metadata(self):
        if not self.specs:
            return (("mode", "estimation-only"),)
        return tuple((capability, spec.result_schema)
                     for capability, spec in self.specs.items())


class ProgramBackend(Backend):
    """Import CUDA-Q / Quake input to portable CUDA-Q Logical P0."""

    # CUDA-Q Logical lowers the MLIR artifact itself, so the local QIR/LLVM JIT artifact
    # would only be built to be thrown away.
    supports_jit = False

    def __init__(self, *, next_backend):
        super().__init__(LoweringSpec((),
                                      None,
                                      accepted_stages=("CUDA-Q / Quake",),
                                      produced_stage="p0"),
                         next_backend=next_backend)

    def compile(self, source, *, arguments=(), **_options):
        from ..compiler import Build, import_cudaq
        from ..compiler.quake import import_quake
        if isinstance(source, Build):
            return source
        if hasattr(source, "mlir_module"):
            return import_quake(str(source.mlir_module))
        return import_cudaq(source, *tuple(arguments or ()))

    def stack_metadata(self):
        return (("source", "CUDA-Q / Quake"),)


class CliffordTBackend(Backend):
    """Legalize portable P0 programs to the positive H/S/T/CX gate set."""

    def __init__(self, *, precision=1.0e-4, next_backend):
        # Let the gate-set own precision validation so this backend and the
        # public ``cudaq.logical.compiler.synthesize`` entry point have identical rules.
        from ..compiler.gate_sets import clifford_t

        self.precision = dict(
            clifford_t.pipeline(
                precision=precision).passes[0].options)["precision"]
        super().__init__(LoweringSpec((),
                                      None,
                                      accepted_stages=("p0",),
                                      produced_stage="p0"),
                         next_backend=next_backend)

    def compile(self, build, **_options):
        from ..compiler import Build

        if not isinstance(build, Build):
            raise TypeError(
                "CliffordTBackend expects a cudaq.logical.compiler.Build")
        if build.profile != "p0":
            raise ValueError("CliffordTBackend accepts only P0 Builds, "
                             f"got {build.profile!r}")
        synthesis = build.synthesis
        if synthesis is not None and synthesis.gate_set == "clifford_t":
            return build

        from ..compiler.gate_sets import clifford_t
        from ..compiler.synthesize import synthesize

        return synthesize(build, gate_set=clifford_t, precision=self.precision)

    def stack_metadata(self):
        return (("gate set", "H/S/T/CX"), ("precision", f"{self.precision:g}"))


def _profile_rank(build):
    return {
        "p0": 0,
        "p1": 1,
        "p2a": 2,
        "p2n": 2,
        "p3": 3,
        "p4": 4
    }[build.profile]


def _estimate_tier(tier):
    from .. import estimate

    if isinstance(tier, str):
        try:
            tier = estimate.Tier[tier.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown estimation tier {tier!r}") from exc
    if not isinstance(tier, estimate.Tier):
        raise TypeError(
            "tier must be a cudaq.logical.estimate.Tier or tier name")
    return tier


def _stage_estimate(build, *, tier=None, **estimate_options):
    """Return the estimate tier naturally owned by one accepted build."""

    from .. import estimate
    from ..compiler import Build

    if not isinstance(build, Build):
        return {}
    tier = _estimate_tier(tier) if tier is not None else None
    if build.profile == "p0":
        if tier is not None and tier is not estimate.Tier.LOGICAL:
            return {}
        return {
            estimate.Tier.LOGICAL.name:
                estimate(build, tier=estimate.Tier.LOGICAL)
        }
    if build.profile in {"p2a", "p2n"}:
        if tier is not None and tier is not estimate.Tier.STATIC:
            return {}
        return {
            estimate.Tier.STATIC.name:
                estimate(build, tier=estimate.Tier.STATIC)
        }
    return {}


def _merge_estimates(own, downstream):
    """Merge distinct tier maps, preferring the downstream value on collision."""

    return {**own, **downstream}


def _cudaq_estimate_result(estimates):
    from cudaq import EstimateResult

    return EstimateResult(annotations={
        name: value.to_dict() for name, value in estimates.items()
    })


class LogicalMachineBackend(Backend):

    def __init__(self, architecture, *, next_backend):
        super().__init__(LoweringSpec((),
                                      None,
                                      accepted_stages=("p0",),
                                      produced_stage="p1"),
                         next_backend=next_backend)
        self.architecture = architecture

    def compile(self, build, **options):
        if _profile_rank(build) >= 1:
            return build
        from ..compiler import compile, pipelines
        return compile(build,
                       pipeline=pipelines.placed(),
                       device=self.architecture,
                       placement=options.get("placement", ()),
                       constraints=options.get("constraints"),
                       objective=options.get("objective"))

    def stack_metadata(self):
        entries = [("machine", self.architecture.name)]
        entries.extend(
            (space.name, "capacity=" +
             ("unbounded" if space.capacity is None else str(space.capacity)))
            for space in self.architecture.spaces)
        return tuple(entries)


class QECMachineBackend(Backend):

    def __init__(self, machine, *, next_backend):
        super().__init__(LoweringSpec((),
                                      None,
                                      accepted_stages=("p1",),
                                      produced_stage="p2"),
                         next_backend=next_backend)
        self.machine = machine

    def compile(self, build, **options):
        if _profile_rank(build) >= 2:
            return build
        from ..compiler import compile, pipelines
        return compile(build,
                       pipeline=pipelines.qec(),
                       device=self.machine,
                       policy=options.get("policy"))

    def stack_metadata(self):
        device = self.machine
        qec = getattr(device, "qec", device)
        entries = [("machine", qec.name)]
        bindings = {
            binding.qec_region.name: binding
            for binding in getattr(device, "logical_to_qec", ())
        }
        for region in qec.regions:
            binding = bindings.get(region.name)
            architecture = (binding.architecture.name if binding is not None and
                            binding.architecture is not None else
                            "default encoding")
            code, distance = region.encoding.code, region.encoding.code.d
            description = f"code={code.name}"
            if distance.value is not None:
                description += f", d={distance.value}"
            entries.append(
                (region.name, f"{description}, blocks={region.block_capacity}, "
                 f"architecture={architecture}"))
        return tuple(entries)


class PhysicalMachineBackend(Backend):

    def __init__(self, machine, *, next_backend):
        super().__init__(LoweringSpec((),
                                      None,
                                      accepted_stages=("p2",),
                                      produced_stage="p3"),
                         next_backend=next_backend)
        self.machine = machine

    def compile(self, build, **_options):
        if _profile_rank(build) >= 3:
            return build
        from ..compiler import compile, pipelines
        return compile(build,
                       pipeline=pipelines.physical(),
                       device=self.machine)

    def stack_metadata(self):
        return (("machine", self.machine.name),)


class Target(CustomTarget):
    """Data-defined CUDA-Q target with one recipe per real capability."""

    name = "unknown"
    _specs = MappingProxyType({})
    available = True
    _plugin = None
    _plugin_version = None
    _availability = "local_optional_runtime"

    @staticmethod
    def _new_compile_target():
        """Create the CUDA-Q compile configuration owned by one target."""

        from ..compiler.quake import CUDAQ_TO_P0_PREPARATION_PIPELINE

        target = CompileTarget()
        target.fully_specialize = True
        target.support_resource_counts = False
        target.support_conditionals_on_measure_results = False
        target.pipeline_config.override_pass_pipeline = CUDAQ_TO_P0_PREPARATION_PIPELINE
        return target

    @classmethod
    def _create(cls,
                name,
                specs,
                runtime_endpoint,
                *,
                plugin=None,
                version=None,
                availability="local_optional_runtime"):
        """Construct one fully configured CUDA-Q / QLX target instance."""

        methods = {
            capability: _capability_method(capability) for capability in specs
        }
        target_type = type(
            "".join(piece.title() for piece in name.split("_")) + "Target",
            (cls,),
            methods,
        )
        target_type.name = str(name)
        instance = target_type(runtime_endpoint=runtime_endpoint,
                               compile_target=cls._new_compile_target())
        instance._specs = MappingProxyType(dict(specs))
        instance._plugin = None if plugin is None else str(plugin)
        instance._plugin_version = None if version is None else str(version)
        instance._availability = str(availability)
        return instance

    @classmethod
    def define(
        cls,
        name,
        *,
        emit_text=None,
        emit_circuit=None,
        emit_artifact=None,
        plugin=None,
        version=None,
        availability="local_optional_runtime",
    ):
        specs = {
            "emit_text": emit_text,
            "emit_circuit": emit_circuit,
            "emit_artifact": emit_artifact,
        }
        if emit_circuit is not None and emit_text is None:
            specs["emit_text"] = emit_circuit
        specs = {
            key: value for key, value in specs.items() if value is not None
        }
        if any(not isinstance(spec, LoweringSpec) for spec in specs.values()):
            raise TypeError("target capabilities must be LoweringSpec values")

        specs = MappingProxyType(dict(specs))
        return cls._create(name,
                           specs,
                           ProgramBackend(next_backend=TerminalBackend(specs)),
                           plugin=plugin,
                           version=version,
                           availability=availability)

    @classmethod
    def from_backend(cls,
                     name,
                     backend,
                     *,
                     plugin=None,
                     version=None,
                     availability="local_optional_runtime"):
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a qlx.targets.Backend")
        return cls._create(name, {},
                           backend,
                           plugin=plugin,
                           version=version,
                           availability=availability)

    @classmethod
    def _from_stack(cls,
                    name,
                    *,
                    architecture,
                    device,
                    runtime_backend,
                    preprocess=None,
                    source_modules: Iterable[str] = (),
                    plugin=None,
                    version=None,
                    availability="local_optional_runtime"):
        """Create a target from normalized logical architecture and device data."""

        from ..architecture import LogicalMachine
        from ..devices import Device

        if not isinstance(architecture, LogicalMachine):
            raise TypeError(
                "Target architecture must be a cudaq.logical.architecture.LogicalMachine"
            )
        if device is not None and not isinstance(device, Device):
            raise TypeError(
                "Target device must be a cudaq.logical.devices.Device")
        if not isinstance(runtime_backend, Target):
            raise TypeError("runtime_backend must be a Target")
        backend = runtime_backend.runtime_endpoint
        # Targets are themselves exposed through ProgramBackend -> terminal.
        # A composed target already has a ProgramBackend at its outer edge, so
        # splice this one-level wrapper rather than reporting or traversing it
        # as a redundant second P0 import stage.
        if (isinstance(backend, ProgramBackend) and
                isinstance(backend.next_backend, TerminalBackend)):
            backend = backend.next_backend
        if device is not None and getattr(device, "physical", None) is not None:
            backend = PhysicalMachineBackend(device, next_backend=backend)
        if device is not None and device.qec is not None:
            backend = QECMachineBackend(device, next_backend=backend)
        backend = LogicalMachineBackend(architecture, next_backend=backend)
        if preprocess is not None:
            backend = preprocess(backend)
            if not isinstance(backend, Backend):
                raise TypeError("Target preprocess must return a Backend")
        target = cls._create(name,
                             runtime_backend._specs,
                             ProgramBackend(next_backend=backend),
                             plugin=plugin,
                             version=version,
                             availability=availability)
        target._architecture, target._device = architecture, device
        target._source_modules = tuple(dict.fromkeys(source_modules))
        return target

    @classmethod
    def from_device(cls, name, device, **kwargs):
        """Create a target that lowers through the device's configured layers."""

        from ..devices import Device

        if not isinstance(device, Device):
            raise TypeError(
                "Target device must be a cudaq.logical.devices.Device")
        return cls._from_stack(name,
                               architecture=device.logical,
                               device=device,
                               **kwargs)

    @classmethod
    def from_architecture(cls, name, architecture, **kwargs):
        """Create a target that lowers through a logical architecture."""

        return cls._from_stack(name,
                               architecture=architecture,
                               device=None,
                               **kwargs)

    def capabilities(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def pipelines(self):
        return {
            capability: tuple(stage.describe() for stage in spec.stages)
            for capability, spec in self._specs.items()
        }

    def print_stack(self) -> None:
        """Print this target's configured machines and runtime handoff path."""

        print(
            f"== {self.name} :: CUDA-Q Logical backend stack ====================="
        )
        backend = self.runtime_endpoint
        while backend is not None and not isinstance(backend, TerminalBackend):
            accepted = ", ".join(backend.spec.accepted_stages) or "input"
            stage = (f"{accepted.upper()} -> "
                     f"{backend.spec.produced_stage.upper()}")
            print(f"---- {stage} " + "-" * (52 - len(stage)))
            print(f"  {type(backend).__name__}")
            for key, value in backend.stack_metadata():
                print(f"    {key}: {value}")
            backend = backend.next_backend
        print("---- launch policies " + "-" * 39)
        policies = policies_supported_by(self)
        if not policies:
            print("  (none)")
        for policy in policies:
            print(f"  {policy.__name__}")

    def manifest(self):
        plugin = None
        if self._plugin is not None:
            plugin = {
                "entry_point": self._plugin,
                "version": self._plugin_version,
            }
        return {
            "schema": "qlx.target-manifest/v2",
            "name": self.name,
            "capabilities": self.capabilities(),
            "availability": self._availability,
            "plugin": plugin,
            "recipes": {
                capability: {
                    "accepted_stages": spec.accepted_stages,
                    "required_facets": spec.required_facets,
                    "produced_stage": spec.produced_stage,
                    "provides_facets": spec.provides_facets,
                    "stages": tuple(stage.describe() for stage in spec.stages),
                    "finalizer": _finalizer_identity(spec.finalize),
                    "result_schema": spec.result_schema,
                    "effect": spec.effect,
                } for capability, spec in self._specs.items()
            },
        }

    def serialize(self) -> bytes:
        """Return a canonical portable target-manifest payload."""

        return json.dumps(self.manifest(),
                          sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    @classmethod
    def replay(cls, payload):
        if isinstance(payload, (bytes, bytearray, memoryview)):
            manifest = json.loads(bytes(payload).decode("utf-8"))
        elif isinstance(payload, str):
            manifest = json.loads(payload)
        elif isinstance(payload, dict):
            manifest = payload
        else:
            raise TypeError(
                "Target.replay expects bytes, JSON text, or a mapping")
        if manifest.get("schema") != "qlx.target-manifest/v2":
            raise ValueError("unsupported target manifest schema")
        plugin = manifest.get("plugin")
        if not plugin or not plugin.get("entry_point"):
            raise UnavailableTargetError(
                f"target {manifest.get('name')!r} has no versioned replay provider"
            )
        try:
            module_name, symbol = plugin["entry_point"].split(":", 1)
            resolved = getattr(importlib.import_module(module_name), symbol)
        except (ValueError, ImportError, AttributeError) as exc:
            raise UnavailableTargetError(
                f"target provider {plugin['entry_point']!r} is unavailable"
            ) from exc
        if not isinstance(resolved, Target) and callable(resolved):
            resolved = resolved(manifest)
        if not isinstance(resolved, Target):
            raise UnavailableTargetError(
                f"target provider {plugin['entry_point']!r} did not resolve a Target"
            )
        expected = json.loads(resolved.serialize().decode("utf-8"))
        if expected != manifest:
            raise ValueError(
                "resolved target provider does not match the serialized manifest"
            )
        return resolved

    def materialize(self, module=None):
        from ..compiler.target_manifest import materialize_target

        return materialize_target(self, module=module)

    def __repr__(self):
        return f"<Target {self.name} capabilities={list(self.capabilities())}>"


def _capability_method(capability):

    def method(self, build, **kwargs):
        return _execute(self, capability, build, **kwargs)

    method.__name__ = capability
    return method


def _execute(target, capability, build, _policy_token=None, /, **kwargs):
    try:
        spec = target._specs[capability]
    except KeyError as exc:
        from ..errors import UnsupportedCombination

        raise UnsupportedCombination(
            f"target {target.name!r} does not support {capability}") from exc
    requested_root = kwargs.pop("root_symbol", build.root.symbol)
    if requested_root != build.root.symbol:
        raise ValueError(
            f"target execution root @{requested_root} does not match immutable "
            f"Build root @{build.root.symbol}")
    kwargs["root_symbol"] = build.root.symbol
    operation = "emit_artifact" if capability == "emit_artifact" else "emit"
    return getattr(target.runtime_endpoint, operation)(build,
                                                       capability=capability,
                                                       **kwargs)


def _coerce_build(root):
    from ..compiler import Build, compile

    if isinstance(root, Build):
        return root
    analyzed = getattr(root, "build", None)
    if isinstance(analyzed, Build):
        return analyzed
    return compile(root)


def _require_targetable(build) -> None:
    """Reject a resource surrogate before target-specific lowering."""

    definition = build.definitions.get(build.root.symbol)
    if definition is not None and "estimate_only" in definition.op.attributes:
        from ..errors import EstimateOnlyTargetError

        raise EstimateOnlyTargetError(
            f"estimate-only @{build.root.symbol} cannot be emitted; use "
            "Build.to_mlir() for inspection or "
            "cudaq.logical.analysis.estimate() for resource analysis")


def launch(root, *, target=None, policy=None, **kwargs):
    from ..errors import UnsupportedCombination

    build = _coerce_build(root)
    if target is not None:
        _require_targetable(build)
    if target is None:
        if policy is not None:
            raise TypeError(
                "cudaq.logical.targets.launch policy= requires an explicit target="
            )
        if kwargs:
            raise TypeError(
                "target-independent cudaq.logical.estimate(build) takes no target options"
            )
        # With neither target nor policy, launch requests target-independent
        # Tier-1 static estimation (spec 07 SS7).
        from .. import estimate

        return estimate(build)
    if policy is None:
        capabilities = target.capabilities()
        if "emit_text" in capabilities:
            policy = EmitPolicy()
        else:
            raise UnsupportedCombination(
                f"target {target.name!r} has no inferable capability")
    capability = _POLICY_CAPABILITY.get(type(policy))
    if capability is None:
        raise TypeError(f"unsupported launch policy {type(policy).__name__}")
    policy_token = None
    result = _execute(target, capability, build, policy_token, **kwargs)
    return result


def emit(root, *, target):
    return launch(root, target=target, policy=EmitPolicy())


def emit_artifact(root, *, target):
    """Emit one target's typed artifact instead of its text convenience."""

    return launch(root, target=target, policy=ArtifactPolicy())


def emit_policy():
    return EmitPolicy()


def artifact_policy():
    return ArtifactPolicy()


def policies_supported_by(target):
    inverse = {value: key for key, value in _POLICY_CAPABILITY.items()}
    policies = []
    for capability in target.capabilities():
        # Derived capabilities without their own policy spelling (such as
        # ``emit_circuit``, which is addressed through the generic emit
        # policy) do not add a second launch-policy entry.
        policy = inverse.get(capability)
        if policy is not None and policy not in policies:
            policies.append(policy)
    return tuple(policies)


mlir = Target.define(
    "mlir",
    plugin="cudaq.logical.targets:mlir",
    version="0.1",
    emit_text=LoweringSpec((),
                           emit_mlir(),
                           result_schema="text/mlir",
                           accepted_stages=()),
)

# Canonical estimation-only target. It deliberately has no execution
# capabilities and terminates after the target's compilation layers.
estimator = Target.from_backend(
    "estimator",
    ProgramBackend(next_backend=TerminalBackend()),
)


def clifford_t_target(*, precision=1.0e-4):
    """Build an estimation-only target that first legalizes to Clifford+T."""

    return Target.from_backend(
        "clifford_t",
        ProgramBackend(next_backend=CliffordTBackend(
            precision=precision, next_backend=TerminalBackend())),
    )


# Default Clifford+T target. Use ``clifford_t_target(precision=...)`` when a
# different default approximation bound is required.
clifford_t = clifford_t_target()


def qec_target(name,
               *,
               code=None,
               architecture=None,
               logical_capacity=3,
               source_modules=(),
               next_backend=None,
               _preprocess=None):
    """Build a QEC target whose stack currently terminates in ``estimator``.

    This preview carries P1/P2 ``Device`` definitions only. A later physical
    target layer can replace the terminal target without changing the device
    construction or this public factory.
    """

    from .. import devices

    if (code is None) == (architecture is None):
        raise TypeError(
            "qec_target requires exactly one of code= or architecture=")

    if next_backend is None:
        runtime_backend = estimator
    else:
        if not isinstance(next_backend, Backend):
            raise TypeError("qec_target next_backend must be a "
                            "cudaq.logical.targets.Backend or None")
        # ``Target._from_stack`` owns insertion of the device layers. Wrap an
        # explicit tail only to reuse that composition path without adding a
        # second ProgramBackend.
        runtime_backend = Target.from_backend(f"{name}_terminal", next_backend)

    builder = devices.DeviceBuilder(
        f"{name}Device",
        source_module=None,
    )
    compute = builder.logical.add_compute(capacity=logical_capacity)
    if architecture is None:
        builder.qec.bind(compute, encoding=code)
    else:
        builder.qec.bind(compute, architecture=architecture)
    return Target.from_device(
        name,
        builder.build(),
        runtime_backend=runtime_backend,
        preprocess=_preprocess,
        source_modules=source_modules,
    )


def surface_target(*,
                   distance=3,
                   logical_capacity=3,
                   precision=1.0e-4,
                   next_backend=None):
    """Build a surface-code target that legalizes to Clifford+T before QEC."""

    from . import recipes

    return qec_target(
        "surface",
        architecture=recipes.surface_architecture(distance),
        logical_capacity=logical_capacity,
        next_backend=next_backend,
        _preprocess=lambda backend: CliffordTBackend(precision=precision,
                                                     next_backend=backend),
    )


def steane_target(*, logical_capacity=3, next_backend=None):
    """Build a Steane-code QEC target with an estimation-only terminal."""

    from . import recipes

    return qec_target(
        "steane",
        architecture=recipes.steane_architecture(),
        logical_capacity=logical_capacity,
        next_backend=next_backend,
    )


__all__ = [
    "Target",
    "Backend",
    "TerminalBackend",
    "ProgramBackend",
    "CliffordTBackend",
    "LogicalMachineBackend",
    "QECMachineBackend",
    "PhysicalMachineBackend",
    "LoweringSpec",
    "emit_mlir",
    "UnavailableTargetError",
    "EmitPolicy",
    "ArtifactPolicy",
    "launch",
    "emit",
    "emit_artifact",
    "emit_policy",
    "artifact_policy",
    "policies_supported_by",
    "mlir",
    "estimator",
    "clifford_t",
    "clifford_t_target",
    "qec_target",
    "surface_target",
    "steane_target",
]
