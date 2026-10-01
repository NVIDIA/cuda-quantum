# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math

from cudaq.mlir.ir import MLIRError
from ._fabric_walk import (
    _attr_text,
    _children,
    _symbol,
    _walk_executable,
)
from ._parameters import resolve_physical_parameters
from .static import count
from .types import (
    EvidencePolicy,
    FabricEstimate,
    FailureBudget,
    MissingEvidence,
    RetryDemand,
)


def _independent_failure(probability: float, trials: int | float) -> float:
    """Stably aggregate independent equal-probability failure trials."""

    if probability <= 0.0 or trials <= 0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return -math.expm1(float(trials) * math.log1p(-probability))


def _independent_union(*probabilities: float) -> float:
    """Stably combine independent failure probabilities."""

    if any(probability >= 1.0 for probability in probabilities):
        return 1.0
    return -math.expm1(
        sum(
            math.log1p(-probability)
            for probability in probabilities
            if probability > 0.0))


def analytical(
    build,
    *,
    p_phys=None,
    failure_budget,
    scaling=None,
    cycle_time=None,
    evidence_policy=None,
):
    """Tier-2 estimate tied directly to the selected P2 operation network.

    Physical parameters default to the selected device's operating point.
    Explicit values override those defaults for sensitivity studies.
    """
    p_phys, scaling, cycle_time = resolve_physical_parameters(
        build,
        p_phys=p_phys,
        scaling=scaling,
        cycle_time=cycle_time,
    )
    if not isinstance(failure_budget, FailureBudget):
        failure_budget = FailureBudget(float(failure_budget))
    evidence_policy = evidence_policy or EvidencePolicy()
    p_phys = float(p_phys)
    cycle_time = float(cycle_time)
    if not math.isfinite(p_phys) or not 0.0 <= p_phys <= 1.0:
        raise ValueError("p_phys must lie in [0, 1]")
    if not math.isfinite(cycle_time) or cycle_time <= 0.0:
        raise ValueError("cycle_time must be finite and positive")

    counts = count(build)
    # Reparse the immutable Build snapshot instead of trusting the cached
    # inspection module, whose underlying MLIR operation is mutable.
    module = build._fresh_module()
    symbols = {
        _symbol(operation.operation): operation.operation
        for operation in module.body.operations
        if _symbol(operation.operation) is not None
    }
    root = symbols[build.root.symbol]
    if root.name == "fabric.gadget_profile":
        root = symbols[_attr_text(root.attributes["gadget"])]

    def _dictionary(attribute):
        if attribute is None:
            return {}
        return {str(named.name): named.attr for named in attribute}

    def _symbol_path(attribute):
        return tuple(member.strip().strip('"').lstrip("@")
                     for member in str(attribute).split("::"))

    def _has_exact_lvm_key(attribute, kind, key):
        if attribute is None:
            return False
        expected = f'#lvm.{kind}<"{key}">'
        return any(str(value) == expected for value in attribute)

    def _selected_device_operation():
        devices = [
            operation.operation
            for operation in module.body.operations
            if operation.operation.name == "qlx.device"
        ]
        root_name = None
        metadata = root.attributes.get("metadata")
        if metadata is not None:
            try:
                root_name = _attr_text(metadata["device"])
            except (KeyError, TypeError):
                pass
        experiment_device = getattr(getattr(build, "experiment", None),
                                    "device", None)
        experiment_name = None
        if experiment_device is not None:
            experiment_name = (experiment_device.get("name") if isinstance(
                experiment_device, Mapping) else getattr(
                    experiment_device, "name", None))
            if not isinstance(experiment_name, str) or not experiment_name:
                raise MissingEvidence(
                    "Tier.ANALYTICAL experiment has an explicit device binding "
                    "without a stable name")
        explicit_names = {
            name for name in (root_name, experiment_name) if name is not None
        }
        if len(explicit_names) > 1:
            raise MissingEvidence(
                "Tier.ANALYTICAL root and experiment select different devices")
        if explicit_names:
            name = explicit_names.pop()
            exact = [
                candidate for candidate in devices if _symbol(candidate) == name
            ]
            if len(exact) != 1:
                raise MissingEvidence(
                    f"Tier.ANALYTICAL explicit selected device @{name} is "
                    "missing or ambiguous")
            return exact[0]
        if not devices:
            return None
        if len(devices) == 1:
            return devices[0]
        raise MissingEvidence(
            "Tier.ANALYTICAL cannot identify one selected device manifest")

    selected_device = _selected_device_operation()
    selected_logical_path = (() if selected_device is None else _symbol_path(
        selected_device.attributes["logical"]))
    if selected_device is not None and len(selected_logical_path) != 1:
        raise MissingEvidence(
            "selected device must name exactly one logical domain")

    def _selected_manifest(name, expected_operation):
        if selected_device is None or name not in selected_device.attributes:
            raise MissingEvidence(
                f"selected device has no {name.replace('_', '-')} manifest")
        path = _symbol_path(selected_device.attributes[name])
        if len(path) != 1:
            raise MissingEvidence(
                f"selected device {name.replace('_', '-')} must be a top-level symbol"
            )
        operation = symbols.get(path[0])
        if operation is None or operation.name != expected_operation:
            raise MissingEvidence(
                f"selected device has no valid {name.replace('_', '-')} manifest"
            )
        return operation

    def _factory_physical_qubits(factory_name):
        """Resolve one factory's exclusive physical-qubit home from the IR."""

        logical_to_qec = _selected_manifest("logical_to_qec",
                                            "qlx.logical_to_qec")
        if _symbol_path(logical_to_qec.attributes["logical"]) != (
                selected_logical_path):
            raise MissingEvidence(
                "selected logical-to-QEC manifest names another logical domain")
        logical_entries = [
            _dictionary(entry)
            for entry in logical_to_qec.attributes["entries"]
            if _attr_text(_dictionary(entry).get("logical")) == factory_name
        ]
        if len(logical_entries) != 1:
            raise MissingEvidence(
                f"factory @{factory_name} has no unique logical-to-QEC binding")
        qec_name = _attr_text(logical_entries[0]["qec"])

        qec_to_physical = _selected_manifest("qec_to_physical",
                                             "qlx.qec_to_physical")
        qec_machine = _selected_manifest("qec", "fabric.machine")
        physical_machine = _selected_manifest("physical", "phys.machine")
        selected_qec_path = (_symbol(qec_machine),)
        selected_physical_path = (_symbol(physical_machine),)
        if (len(selected_qec_path) != 1 or _symbol_path(
                qec_to_physical.attributes["qec"]) != selected_qec_path or
                len(selected_physical_path) != 1 or
                _symbol_path(qec_to_physical.attributes["physical"])
                != selected_physical_path):
            raise MissingEvidence(
                "selected QEC-to-physical manifest names another device stack")
        physical_entries = [
            _dictionary(entry)
            for entry in qec_to_physical.attributes["entries"]
        ]
        matches = [
            entry for entry in physical_entries
            if _attr_text(entry.get("qec")) == qec_name
        ]
        if len(matches) != 1:
            raise MissingEvidence(
                f"factory @{factory_name} QEC region @{qec_name} has no unique "
                "physical binding")
        resource_names = tuple(
            _attr_text(resource) for resource in matches[0]["resources"])
        if not resource_names:
            raise MissingEvidence(
                f"factory @{factory_name} has no bound physical resources")
        resource_uses = Counter(
            _attr_text(resource)
            for entry in physical_entries
            for resource in entry["resources"])
        shared = tuple(
            name for name in resource_names if resource_uses[name] != 1)
        if shared:
            raise MissingEvidence(
                f"factory @{factory_name} physical resources {shared!r} are "
                "shared or ambiguously bound")

        resource_classes = {
            _symbol(child): child
            for child in _children(physical_machine)
            if child.name == "phys.resource_class"
        }
        missing = tuple(
            name for name in resource_names if name not in resource_classes)
        if missing:
            raise MissingEvidence(
                f"factory @{factory_name} has unresolved physical resources "
                f"{missing!r}")
        bound_qubits = 0
        for name in resource_names:
            resource_class = resource_classes[name]
            kind = _attr_text(resource_class.attributes["kind"])
            unit_kind = _attr_text(
                resource_class.attributes.get("physical_unit_kind", kind))
            if unit_kind != "qubit":
                continue
            count = int(resource_class.attributes["count"])
            units = int(resource_class.attributes.get("physical_units", 1))
            bound_qubits += count * units
        if bound_qubits <= 0:
            raise MissingEvidence(
                f"factory @{factory_name} has no positive physical-qubit home")
        return bound_qubits, matches[0]

    def _artifact_resource_bindings(requested_streams):
        if (selected_device is None or
                "resource_bindings" not in selected_device.attributes):
            return {}
        models = {}
        for raw in selected_device.attributes["resource_bindings"]:
            binding = _dictionary(raw)
            stream_path = _symbol_path(binding["stream"])
            if stream_path not in requested_streams:
                continue
            if stream_path[:-1] != selected_logical_path:
                raise MissingEvidence(
                    f"resource stream {binding['stream']} is outside the "
                    "selected device logical domain")
            domain = symbols.get(stream_path[0])
            if domain is None or domain.name != "lvm.domain":
                raise MissingEvidence(
                    f"resource stream {binding['stream']} has no logical domain"
                )
            stream = next(
                (child for child in _children(domain)
                 if child.name == "lvm.stream" and
                 _symbol(child) == stream_path[-1]),
                None,
            )
            if stream is None:
                raise MissingEvidence(
                    f"resource binding has no stream {binding['stream']}")
            stream_kind = _attr_text(stream.attributes["produces"])
            producer_name = _attr_text(binding["producer"])
            producer = symbols.get(producer_name)
            if producer is None or producer.name != "fabric.protocol":
                raise MissingEvidence(
                    f"resource binding has no producer @{producer_name}")
            capacity = None
            factory_name = None
            bound_qubits = None
            physical_binding = {}
            if "factory" in binding:
                factory_path = _symbol_path(binding["factory"])
                if factory_path[:-1] != selected_logical_path:
                    raise MissingEvidence(
                        f"resource binding factory {binding['factory']} is "
                        "outside the selected device logical domain")
                factory_name = "::".join(factory_path)
                factory = next(
                    (child for child in _children(domain)
                     if child.name == "lvm.space" and
                     _symbol(child) == factory_path[-1]),
                    None,
                )
                if factory is None:
                    raise MissingEvidence(
                        f"resource binding has no factory {binding['factory']}")
                if not _has_exact_lvm_key(
                        factory.attributes.get("capabilities"),
                        "capability",
                        "qlx.machine/logical_factory",
                ):
                    raise MissingEvidence(
                        f"resource binding factory {binding['factory']} lacks "
                        "logical_factory capability")
                has_supply = any(child.name == "lvm.channel" and _symbol_path(
                    child.attributes["from"])[-1] == factory_path[-1] and
                                 _symbol_path(child.attributes["to"])[-1] ==
                                 stream_path[-1] and _has_exact_lvm_key(
                                     child.attributes.get("capabilities"),
                                     "capability",
                                     "qlx.machine/resource_transfer",
                                 ) for child in _children(domain))
                if not has_supply:
                    raise MissingEvidence(
                        f"resource binding factory {binding['factory']} does "
                        "not supply its stream")
                if "capacity" in factory.attributes:
                    capacity = int(factory.attributes["capacity"])
                bound_qubits, physical_binding = _factory_physical_qubits(
                    factory_path[-1])
            metadata = {
                key: _attr_text(value) for key, value in _dictionary(
                    producer.attributes.get("metadata")).items()
            }
            interval = physical_binding.get("factory_output_interval_cycles")
            policy = physical_binding.get("factory_model_policy")
            if interval is not None and policy is not None:
                source_units = physical_binding.get(
                    "factory_source_physical_units")
                if source_units is None:
                    if (capacity is None or capacity <= 0 or
                            bound_qubits is None or
                            bound_qubits % capacity != 0):
                        raise MissingEvidence(
                            f"factory @{factory_name} declared model has no "
                            "unambiguous per-lane physical footprint")
                    per_lane_units = bound_qubits // capacity
                    mode = "declared"
                else:
                    per_lane_units = int(source_units)
                    mode = "compiled"
                required_analytical = {
                    "factory_mode",
                    "produces",
                    "physical_error_rate",
                    "cycles_per_attempt",
                    "pipeline_depth",
                    "acceptance_probability",
                    "output_infidelity",
                    "physical_qubits",
                }
                if not required_analytical <= set(metadata):
                    metadata = {
                        "factory_mode": mode,
                        "produces": stream_kind,
                        "physical_error_rate": p_phys,
                        "cycles_per_attempt": _attr_text(interval),
                        "pipeline_depth": 1,
                        # A compact P3 model describes the accepted schedule,
                        # not retry probability or output fidelity.  Retain a
                        # safe analytical closure without inventing either.
                        "acceptance_probability": 1.0,
                        "output_infidelity": 1.0,
                        "physical_qubits": per_lane_units,
                    }
            models[stream_path] = (
                metadata,
                capacity,
                factory_name,
                stream_kind,
                bound_qubits,
            )
        return models

    def _patch_code(type_) -> str | None:
        text = str(type_)
        if not text.startswith("!fabric.patch<@"):
            return None
        return (text[len("!fabric.patch<@"):-1].split(",",
                                                      1)[0].strip().lstrip("@"))

    def _retry_source_metadata(source, source_name):
        """Resolve the authoritative metadata carrier for retry evidence."""

        if source.name == "fabric.gadget":
            spec_attr = source.attributes.get("spec")
            if spec_attr is None:
                raise MissingEvidence(
                    f"retry probability source @{source_name} has no gadget spec"
                )
            spec_name = _attr_text(spec_attr)
            spec = symbols.get(spec_name)
            if spec is None or spec.name != "fabric.gadget_spec":
                raise MissingEvidence(
                    f"retry probability source @{source_name} has no valid gadget spec"
                )
            try:
                valid_spec = spec.verify()
            except MLIRError as error:
                raise MissingEvidence(
                    f"retry probability source @{source_name} has an invalid gadget spec boundary"
                ) from error
            if not valid_spec:
                raise MissingEvidence(
                    f"retry probability source @{source_name} has an invalid gadget spec boundary"
                )
            boundary = source.attributes.get("realization_boundary")
            boundary_values = _dictionary(boundary)
            if boundary is None or any(
                    boundary_values.get(name) != spec.attributes.get(name)
                    for name in ("ports", "flows")):
                raise MissingEvidence(
                    f"retry probability source @{source_name} disagrees with its gadget spec realization boundary"
                )
            if (source.attributes.get("function_type")
                    != spec.attributes.get("function_type")):
                raise MissingEvidence(
                    f"retry probability source @{source_name} disagrees with its gadget spec"
                )
            metadata_source = spec
        elif source.name in {"fabric.protocol", "fabric.gadget_profile"}:
            metadata_source = source
        else:
            raise MissingEvidence(
                f"retry probability source @{source_name} is not an attempt or profile"
            )
        return _dictionary(metadata_source.attributes.get("metadata"))

    def _resource_demand():
        demand = Counter()
        retry_demands = []
        expected_extra_operations = 0.0
        retry_acceptance = 1.0
        retry_logical_error = 0.0
        visits = tuple(
            _walk_executable(
                _children(root),
                symbols,
                call_stack=(build.root.symbol,),
                error_type=MissingEvidence,
            ))
        for visit in visits:
            operation = visit.operation
            if operation.name == "fabric.resource_request":
                stream_path = _symbol_path(operation.attributes["stream"])
                if (selected_logical_path and
                        stream_path[:-1] != selected_logical_path):
                    raise MissingEvidence(
                        f"resource request stream {operation.attributes['stream']} "
                        "is outside the selected device logical domain")
                demand[(
                    _attr_text(operation.attributes["kind"]),
                    stream_path,
                )] += visit.multiplier
        for visit in visits:
            operation = visit.operation
            if operation.name != "fabric.retry":
                continue
            if "attempt" not in operation.attributes:
                raise MissingEvidence(
                    "retry demand requires an explicit named attempt")
            attempt_name = _attr_text(operation.attributes["attempt"])
            attempt = symbols.get(attempt_name)
            if attempt is None or attempt.name not in {
                    "fabric.gadget",
                    "fabric.protocol",
            }:
                raise MissingEvidence(
                    f"retry attempt @{attempt_name} is missing or not executable"
                )
            attempt_visits = tuple(
                _walk_executable(
                    _children(attempt),
                    symbols,
                    call_stack=(attempt_name,),
                    error_type=MissingEvidence,
                ))
            if any(item.operation.name == "fabric.retry"
                   for item in attempt_visits):
                raise MissingEvidence(
                    "nested retry demand is not supported by Tier.ANALYTICAL")
            per_attempt = Counter()
            for item in attempt_visits:
                child = item.operation
                if child.name != "fabric.resource_request":
                    continue
                stream_path = _symbol_path(child.attributes["stream"])
                if (selected_logical_path and
                        stream_path[:-1] != selected_logical_path):
                    raise MissingEvidence(
                        f"retry resource stream {child.attributes['stream']} "
                        "is outside the selected device logical domain")
                per_attempt[(
                    _attr_text(child.attributes["kind"]),
                    stream_path,
                )] += item.multiplier
            if "success_probability" not in operation.attributes:
                if per_attempt:
                    raise MissingEvidence(
                        f"retry @{attempt_name} consumes resources but has no "
                        "certified success_probability")
                continue
            probability = float(operation.attributes["success_probability"])
            maximum = int(operation.attributes["max_attempts"])
            if (not math.isfinite(probability) or
                    not 0.0 < probability <= 1.0 or maximum <= 0):
                raise MissingEvidence(
                    f"retry @{attempt_name} has invalid probability or bound")
            source_attr = operation.attributes.get("success_probability_source")
            evidence_attr = operation.attributes.get(
                "success_probability_evidence")
            if source_attr is None or evidence_attr is None:
                raise MissingEvidence(
                    f"retry @{attempt_name} probability has no typed provenance"
                )
            source_name = _attr_text(source_attr)
            source = symbols.get(source_name)
            if source is None:
                raise MissingEvidence(
                    f"retry @{attempt_name} probability source @{source_name} is missing"
                )
            source_metadata = _retry_source_metadata(source, source_name)
            try:
                established_probability = float(
                    _attr_text(source_metadata["success_probability"]))
                established_evidence = _attr_text(
                    source_metadata["success_probability_evidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MissingEvidence(
                    f"retry @{attempt_name} probability source is not established"
                ) from exc
            evidence = _attr_text(evidence_attr)
            if established_probability != probability or established_evidence != evidence:
                raise MissingEvidence(
                    f"retry @{attempt_name} probability disagrees with its source"
                )

            enclosing = next(
                (symbols[name]
                 for name in reversed(visit.call_stack)
                 if name in symbols and
                 "specialization" in symbols[name].attributes),
                None,
            )
            specialization = _dictionary(
                None if enclosing is
                None else enclosing.attributes.get("specialization"))
            callee_action_site = (None if enclosing is None or "action_site"
                                  not in enclosing.attributes else _attr_text(
                                      enclosing.attributes["action_site"]))
            invocation = next(
                (candidate for candidate in reversed(visit.call_path)
                 if "action_site" in candidate.attributes),
                None,
            )
            call_action_site = (None if invocation is None or "action_site"
                                not in invocation.attributes else _attr_text(
                                    invocation.attributes["action_site"]))
            if (callee_action_site is not None and
                    call_action_site is not None and
                    callee_action_site != call_action_site):
                raise MissingEvidence(
                    f"retry @{attempt_name} has conflicting call/callee "
                    "action-site provenance")
            action_site = call_action_site or callee_action_site
            effective_angle = (None if "effective_angle" not in specialization
                               else float(specialization["effective_angle"]))
            precision = (None if "precision" not in specialization else float(
                specialization["precision"]))
            synthesis_sha256 = None
            synthesis_prefix = "synthesis:sha256:"
            if evidence.startswith(synthesis_prefix):
                synthesis_sha256 = evidence.removeprefix(synthesis_prefix)
                attempt_metadata = _retry_source_metadata(attempt, attempt_name)
                specialization_digest = (
                    None if enclosing is None else _attr_text(
                        specialization.get("synthesis_sha256")))
                if (source_name != attempt_name or
                        _attr_text(attempt_metadata.get("synthesis_sha256"))
                        != synthesis_sha256 or
                    (enclosing is not None and
                     specialization_digest != synthesis_sha256)):
                    raise MissingEvidence(
                        f"retry @{attempt_name} synthesis digest is not bound to "
                        "its selected action site")
            elif evidence.startswith("analysis:"):
                profile_name = _attr_text(operation.attributes.get("profile"))
                if source_name != profile_name or source.name != "fabric.gadget_profile":
                    raise MissingEvidence(
                        f"retry @{attempt_name} analysis evidence is not the "
                        "selected profile")
                profile_gadget = source.attributes.get("gadget")
                if (profile_gadget is None or
                        _attr_text(profile_gadget) != attempt_name):
                    raise MissingEvidence(
                        f"retry @{attempt_name} selected profile does not "
                        "analyze the retry attempt")
            else:
                raise MissingEvidence(
                    f"retry @{attempt_name} uses unsupported probability evidence"
                )

            if probability == 1.0:
                completion = 1.0
                exhaustion = 0.0
            else:
                log_exhaustion = maximum * math.log1p(-probability)
                # Each side needs its own stable evaluation. Recovering either
                # one as the binary64 complement of the other erases the tiny
                # completion tail for p << 1 or the tiny exhaustion tail for
                # p close to one.
                completion = -math.expm1(log_exhaustion)
                exhaustion = math.exp(log_exhaustion)
            expected_attempts = completion / probability
            extra_attempts = visit.multiplier * (expected_attempts - 1.0)
            for key, count_per_attempt in per_attempt.items():
                demand[key] += extra_attempts * count_per_attempt

            per_kind = Counter()
            for (kind, _), count_per_attempt in per_attempt.items():
                per_kind[kind] += count_per_attempt
            expected_by_kind = {
                kind: visit.multiplier * expected_attempts * count_per_attempt
                for kind, count_per_attempt in per_kind.items()
            }
            maximum_by_kind = {
                kind: visit.multiplier * maximum * count_per_attempt
                for kind, count_per_attempt in per_kind.items()
            }
            retry_demands.append(
                RetryDemand(
                    attempt=attempt_name,
                    occurrences=visit.multiplier,
                    success_probability=probability,
                    max_attempts=maximum,
                    expected_attempts=expected_attempts,
                    exhaustion_probability=exhaustion,
                    resource_requests_per_attempt=per_kind,
                    expected_resource_requests=expected_by_kind,
                    maximum_resource_requests=maximum_by_kind,
                    action_site=action_site,
                    effective_angle=effective_angle,
                    precision=precision,
                    synthesis_sha256=synthesis_sha256,
                ))

            attempt_operations = sum(
                item.multiplier
                for item in attempt_visits
                if (item.operation.name.startswith("fabric.") or
                    item.operation.name.startswith("cflow.") or item.operation.
                    name.startswith("event.")) and item.operation.name not in {
                        "fabric.gadget",
                        "fabric.protocol",
                        "fabric.protocol_return",
                        "fabric.return",
                        "fabric.profile_end",
                        "fabric.gadget_profile",
                        "cflow.yield",
                        "event.yield",
                    })
            expected_extra_operations += extra_attempts * attempt_operations
            exhaustion_action = _attr_text(
                operation.attributes.get("exhaustion", "report_failure"))
            if exhaustion_action in {"abort", "report_failure"}:
                retry_acceptance *= completion**visit.multiplier
            elif exhaustion_action == "return_last":
                retry_logical_error = _independent_union(
                    retry_logical_error,
                    _independent_failure(exhaustion, visit.multiplier),
                )
            else:
                raise MissingEvidence(
                    f"retry @{attempt_name} has unsupported exhaustion action "
                    f"{exhaustion_action!r}")
        return (
            demand,
            visits,
            tuple(retry_demands),
            expected_extra_operations,
            retry_acceptance,
            retry_logical_error,
        )

    demand = executable_visits = None
    retry_demands = ()
    expected_extra_operations = 0.0
    retry_acceptance = 1.0
    retry_logical_error = 0.0
    code_names = []
    if root.regions and root.regions[0].blocks:
        for argument in root.regions[0].blocks[0].arguments:
            name = _patch_code(argument.type)
            if name is not None and name not in code_names:
                code_names.append(name)
    if not code_names:
        # Closed roots (a placed/QEC-selected workload) carry no patch-typed
        # arguments; their selected codes appear on allocated patch values
        # inside the body instead.
        (
            demand,
            executable_visits,
            retry_demands,
            expected_extra_operations,
            retry_acceptance,
            retry_logical_error,
        ) = _resource_demand()
        for visit in executable_visits:
            operation = visit.operation
            for result in operation.results:
                name = _patch_code(result.type)
                if name is not None and name not in code_names:
                    code_names.append(name)
    if not code_names:
        raise MissingEvidence(
            "Tier.ANALYTICAL needs an executable P2 root with typed code boundaries"
        )

    distances = []
    statuses = []
    physical_qubits = 0
    for code_name in code_names:
        code = symbols.get(code_name)
        if code is None or code.name != "fabric.code":
            raise MissingEvidence(f"missing code definition @{code_name}")
        distance = int(code.attributes["distance"])
        metadata = code.attributes.get("metadata")
        status = "claimed"
        if metadata is not None:
            try:
                status = _attr_text(metadata["distance_status"])
            except (KeyError, TypeError):
                pass
        if distance <= 0:
            raise MissingEvidence(
                f"code @{code_name} has no usable distance evidence")
        if evidence_policy.require_established and status not in {
                "exact", "lower_bound", "circuit_distance"
        }:
            raise MissingEvidence(
                f"code @{code_name} distance status {status!r} is not established"
            )
        distances.append(distance)
        statuses.append(status)
        physical_qubits += sum(
            int(named.attr) for named in code.attributes["partitions"])

    distance = min(distances)
    site_error = scaling.p_logical(distance, p_phys)
    sites = counts.total_operations + expected_extra_operations
    logical_error = _independent_union(_independent_failure(site_error, sites),
                                       retry_logical_error)
    compute_time = sites * cycle_time
    wallclock = compute_time
    acceptance = retry_acceptance
    bottleneck = "compute_limited"
    assumptions = [
        "independent logical fault sites",
        "serial critical-path upper bound",
        type(scaling).__name__,
    ]

    if demand is None:
        (
            demand,
            executable_visits,
            retry_demands,
            expected_extra_operations,
            retry_acceptance,
            retry_logical_error,
        ) = _resource_demand()
        sites = counts.total_operations + expected_extra_operations
        logical_error = _independent_union(
            _independent_failure(site_error, sites), retry_logical_error)
        compute_time = sites * cycle_time
        wallclock = compute_time
        acceptance = retry_acceptance
    if retry_demands:
        assumptions.extend((
            "certified independent retry-attempt probabilities",
            "bounded retry resource demand includes failed attempts",
        ))
    device = getattr(getattr(build, "experiment", None), "device", None)
    physical = getattr(device, "physical", None)
    artifact_physical = None
    if selected_device is not None and "physical" in selected_device.attributes:
        # Reuse the native analytical pass for the selected device's active
        # physical closure.  Summing ``phys.resource_class`` declarations is
        # not equivalent: patch resources carry multiple physical qubits per
        # member, and an inactive auxiliary patch must not inflate the peak.
        # Tier 3 uses this same verified lower-tier materialization, so this
        # also keeps the public Tier-2 and Tier-3 estimates consistent.
        existing = {
            name for operation in module.body.operations
            if (name := _symbol(operation.operation)) is not None
        }

        def _unique_result_name(base):
            name = base
            while name in existing:
                name += "_"
            existing.add(name)
            return name

        static_result = _unique_result_name("__cudaq_logical_static")
        analytical_result = _unique_result_name("__cudaq_logical_analytical")
        from cudaq.logical._native import native

        try:
            native._materialize_verified_analytical_lower_tier(
                module,
                _symbol(root),
                _symbol(selected_device),
                static_result,
                analytical_result,
                p_phys,
                failure_budget.total,
                cycle_time,
                scaling.prefactor,
                scaling.threshold,
                evidence_policy.require_established,
            )
        except RuntimeError as error:
            raise MissingEvidence(
                "Tier.ANALYTICAL could not authenticate the selected "
                "device's active physical-resource closure") from error
        native_result = next(
            (operation.operation
             for operation in module.body.operations
             if _symbol(operation.operation) == analytical_result),
            None,
        )
        if native_result is None or native_result.name != "qlx.estimate_result":
            raise MissingEvidence(
                "Tier.ANALYTICAL native physical-resource result is missing")
        native_data = _dictionary(native_result.attributes["data"])
        if "physical_qubits_peak" not in native_data:
            raise MissingEvidence(
                "Tier.ANALYTICAL native physical-resource result has no peak")
        artifact_physical = int(native_data["physical_qubits_peak"])
    if artifact_physical is not None:
        physical_qubits = artifact_physical
        assumptions.append("configured physical-device footprint")
    elif physical is not None:
        physical_qubits = sum(resource.count
                              for resource in physical.resource_classes
                              if resource.kind == "qubit")
        assumptions.append("configured physical-device footprint")

    if demand:
        artifact_models = _artifact_resource_bindings(
            {stream_path for _, stream_path in demand})
        logical_name = (selected_logical_path[0] if selected_logical_path else
                        getattr(getattr(device, "logical", None), "name", None))
        streams = {
            (logical_name, stream.name): stream
            for stream in getattr(getattr(device, "logical", None), "streams", (
            ))
        }
        factory_times = Counter()
        resource_error = 0.0
        conservative_factory_error = False
        for (kind, stream_path), requested in sorted(demand.items()):
            stream = streams.get(stream_path)
            producer = None if stream is None else stream.produced_by
            if stream_path in artifact_models:
                (
                    metadata,
                    capacity,
                    factory_name,
                    stream_kind,
                    bound_qubits,
                ) = artifact_models[stream_path]
            else:
                metadata = {} if producer is None else dict(producer.metadata)
                capacity = None if stream is None or stream.region is None else (
                    stream.region.capacity)
                factory_name = (None if stream is None or stream.region is None
                                else stream.region.name)
                stream_kind = (None if stream is None else stream.produces.name)
                bound_qubits = None
            if stream_kind != kind:
                raise MissingEvidence(f"resource request @{kind} uses stream "
                                      f"@{'::@'.join(stream_path)} "
                                      f"that produces @{stream_kind}")
            required = {
                "factory_mode",
                "produces",
                "physical_error_rate",
                "cycles_per_attempt",
                "pipeline_depth",
                "acceptance_probability",
                "output_infidelity",
                "physical_qubits",
            }
            missing = required - set(metadata)
            if missing or (stream_path not in artifact_models and
                           producer is None):
                raise MissingEvidence(
                    f"Tier.ANALYTICAL resource @{kind} from "
                    f"@{'::@'.join(stream_path)} "
                    "requires one backed "
                    "producer with matching kind, physical-error point, "
                    "factory-model cycles, pipeline depth, acceptance, and "
                    "output-infidelity evidence")
            if metadata["factory_mode"] not in {
                    "analytical", "scheduled_macro", "compiled", "declared"
            }:
                raise MissingEvidence(
                    f"resource @{kind} producer has unsupported factory mode "
                    f"{metadata['factory_mode']!r}")
            if metadata["produces"] != kind:
                raise MissingEvidence(f"resource @{kind} producer declares "
                                      f"@{metadata['produces']}")
            try:
                producer_p_phys = float(metadata["physical_error_rate"])
                cycles = float(metadata["cycles_per_attempt"])
                depth = int(metadata["pipeline_depth"])
                probability = float(metadata["acceptance_probability"])
                output_error = float(metadata["output_infidelity"])
                producer_qubits = int(metadata["physical_qubits"])
            except (TypeError, ValueError) as error:
                raise MissingEvidence(
                    f"resource @{kind} producer factory metadata is not numeric"
                ) from error
            if not math.isclose(
                    producer_p_phys,
                    p_phys,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-16,
            ):
                raise MissingEvidence(
                    f"resource @{kind} producer operating point "
                    f"p_phys={producer_p_phys:g} does not match requested "
                    f"p_phys={p_phys:g}")
            if (not math.isfinite(cycles) or cycles <= 0.0 or depth <= 0 or
                    capacity is None or capacity <= 0 or
                    not 0.0 < probability <= 1.0 or
                    not 0.0 <= output_error <= 1.0 or producer_qubits <= 0):
                raise MissingEvidence(
                    f"resource @{kind} producer factory model has invalid "
                    "capacity, timing, acceptance, output error, or footprint")
            if bound_qubits is None:
                raise MissingEvidence(
                    f"resource @{kind} producer has no artifact-backed physical "
                    "factory home")
            required_qubits = capacity * producer_qubits
            if bound_qubits < required_qubits:
                raise MissingEvidence(
                    f"resource @{kind} factory physical home has {bound_qubits} "
                    f"qubits but capacity {capacity} requires at least "
                    f"{required_qubits} for its producer footprint")
            attempts = requested / probability
            factory_times[factory_name] += (attempts * cycles * cycle_time /
                                            (capacity * depth))
            resource_error = _independent_union(
                resource_error,
                _independent_failure(output_error, requested),
            )
            conservative_factory_error |= metadata["factory_mode"] in {
                "compiled", "declared"
            }
        logical_error = _independent_union(logical_error, resource_error)
        factory_time = max(factory_times.values())
        if factory_time > compute_time:
            wallclock = factory_time
            bottleneck = "factory_limited"
        assumptions.extend((
            "steady-state independent factory attempts",
            "producer pipeline depth is concurrent attempt capacity",
            "resource output infidelity is independent",
        ))
        if conservative_factory_error:
            assumptions += (
                "compact factory timing is conditional on an accepted output",
                "factory retry and output-error models are absent; analytical "
                "resource error is conservatively one",
            )
    return FabricEstimate(
        counts=counts,
        p_phys=p_phys,
        failure_budget=failure_budget,
        distance=distance,
        distance_status=(statuses[0] if len(set(statuses)) == 1 else "mixed"),
        logical_error=logical_error,
        physical_qubits_peak=physical_qubits,
        wallclock=wallclock,
        cycle_time=cycle_time,
        acceptance=acceptance,
        bottleneck=bottleneck,
        budget_met=logical_error <= failure_budget.total,
        assumptions=tuple(assumptions),
        retry_demands=retry_demands,
    )
