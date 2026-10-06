# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exercise an independently importable QEC provider across process replay."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

# This ordinary module is written outside the checkout. Its public compiler,
# protocol, projector, and replay validator must survive a fresh interpreter.
_PROVIDER_MODULE = r'''
from __future__ import annotations

from hashlib import sha256

import cudaq.logical as qlx
from cudaq.logical.qec.lattice_surgery import (
    LatticeSurgeryCompiler,
    QECNetworkArtifact,
    QECNetworkEpoch,
    QECNetworkPlan,
    pipeline_sha256,
    provider_replay_validator,
)


CODE = qlx.codes.CSSCode(
    name="external_fixture_code",
    n=1,
    k=1,
    d=1,
    block=qlx.codes.CSSBlock(data=1),
    lx=((0,),),
    lz=((0,),),
)


@qlx.program
def measure_product() -> bool:
    data = qlx.allocate(1, state=qlx.types.zero)
    data[0], outcome = qlx.mpp(qlx.types.Z(data[0]))
    qlx.discard(data)
    return outcome


@qlx.protocol
def realize_epoch(
    block: qlx.patch[CODE],
) -> tuple[qlx.patch[CODE], bool]:
    block, outcome = qlx.mpp(qlx.types.Z(block[0]))
    return block, outcome


REVOKE_REPLAY = False
PROVIDER_KEY = "example.external:single_mpp@1"


def _expected_artifact(site):
    parameters = site["parameters"]
    if parameters["x_mask"] != 0 or parameters["z_mask"] != 1:
        raise ValueError("fixture provider accepts only one Z product")
    return {"action": site["symbol"], "pauli": "Z"}


@provider_replay_validator(PROVIDER_KEY)
def validate_replayed_plan(context):
    expected = _expected_artifact(
        context.request_payload["actions"][0]["site"])
    if (REVOKE_REPLAY or
            dict(context.plan.artifact.payload) != expected):
        raise ValueError("external replay artifact differs from request")


class ExternalCompiler(LatticeSurgeryCompiler,
                       qlx.compiler.PhysicalProjector):
    __slots__ = (
        "_lowering",
        "_projection_pipeline",
        "corrupt_artifact",
        "reject_artifact",
    )

    def __init__(self, *, corrupt_artifact=False):
        super().__init__(
            plugin="example.external",
            version="1",
            name="single_mpp",
            pipeline=qlx.compiler.Pipeline(
                passes=(qlx.compiler.PassSpec("external-materialize-p2"),),
                output_profile="p2n",
            ),
        )
        self._lowering = qlx.compiler.QECLowering(
            compiler=self,
            objective_family="pauli_product_measurement",
            objective=qlx.logical.mpp,
            codes=(CODE.default_encoding,),
            plugin=self.plugin,
            version=self.version,
            name="external_mpp",
        )
        self._projection_pipeline = qlx.compiler.Pipeline(
            passes=(qlx.compiler.PassSpec("external-project-p3"),),
            output_profile="p3",
        )
        self.corrupt_artifact = corrupt_artifact
        self.reject_artifact = False

    @property
    def qec_lowerings(self):
        return (self._lowering,)

    @property
    def replay_validator(self):
        return validate_replayed_plan

    @property
    def projection_pipeline(self):
        return self._projection_pipeline

    def accepts(self, problem, device):
        return True

    def architecture_digest(self, device):
        return "a" * 64

    def plan_network(self, request, context):
        if len(request.actions) != 1:
            raise ValueError("fixture requires one network action")
        site = request.actions[0].site
        artifact = _expected_artifact({
            "symbol": site.symbol,
            "parameters": site.parameters,
        })
        if self.corrupt_artifact:
            artifact["pauli"] = "X"
        return QECNetworkPlan(
            request_sha256=request.digest,
            lowering_manifest_sha256=request.lowering_manifest_sha256,
            device_architecture_sha256=request.device_architecture_sha256,
            policy_sha256=request.policy_sha256,
            provider_key=self.key,
            required_projector_key=self.key,
            required_projector_pipeline_sha256=pipeline_sha256(
                self.projection_pipeline),
            epochs=(QECNetworkEpoch(
                id="epoch0",
                region=request.regions[0].id,
                actions=(site.symbol,),
            ),),
            artifact=QECNetworkArtifact(
                schema="example.external.plan/v1",
                media_type="application/json",
                payload=artifact,
            ),
        )

    def validate_plan_artifact(self, request, plan):
        site = request.actions[0].site
        expected = _expected_artifact({
            "symbol": site.symbol,
            "parameters": site.parameters,
        })
        if self.reject_artifact or dict(plan.artifact.payload) != expected:
            raise ValueError("external provider artifact differs from request")

    def emit_region(self, region, output):
        epoch = region.epochs[0]
        output.realize_epoch(
            epoch,
            realize_epoch,
            placements=region.actions[0].site.placements,
            actions=epoch.actions,
        )

    def accepts_projection(self, source, device):
        return source.profile == "p2n"

    def projection_architecture_digest(self, device):
        return "b" * 64

    def projection_architecture(self, projection, device):
        return qlx.architecture.PhysicalMachine(
            "ExternalFixturePhysicalMachine",
            resource_classes={
                "qubits": qlx.architecture.ResourceClass(
                    "qubit", 1, name="qubits"),
            },
        )

    def emit_projection(self,
                        projection,
                        device,
                        *,
                        builder,
                        pipeline,
                        experiment=None):
        events = builder.events
        resource = next(iter(events.architecture.resource_classes))
        qubit, = events.acquire(resource, count=1)
        qubit, = events.reset((qubit,))
        outcome = events.measure(qubit)
        builder.seal(outcome)
        return builder.finish(evidence=(qlx.compiler.EvidenceRecord(
            kind="external_fixture_projection",
            producer=self.key,
            result="pass",
            obligations=(
                f"source-p2-digest={sha256(projection.source.serialize()).hexdigest()}",
                "projection-architecture-digest="
                f"{self.projection_architecture_digest(device)}",
            ),
        ),))


def device_for(compiler):
    builder = qlx.devices.DeviceBuilder("ExternalFixtureDevice")
    compute = builder.logical.add_compute(capacity=1)
    builder.qec.bind(compute, encoding=CODE)
    builder.add_compiler(compiler)
    return builder.build()
'''


_COMPILE = r'''
from pathlib import Path
import sys

try:
    import _cudaq_logical_devpath
except ModuleNotFoundError:
    pass
import cudaq.logical as qlx
import external_fixture as fixture

bad = fixture.ExternalCompiler(corrupt_artifact=True)
try:
    qlx.compile(
        fixture.measure_product,
        pipeline=qlx.compiler.pipelines.qec(),
        device=fixture.device_for(bad),
    )
except ValueError as error:
    assert "external provider artifact differs" in str(error)
else:
    raise AssertionError("forged provider artifact reached P2")

good = fixture.ExternalCompiler()
p2 = qlx.compile(
    fixture.measure_product,
    pipeline=qlx.compiler.pipelines.qec(),
    device=fixture.device_for(good),
)
assert p2.profile == "p2n"
Path(sys.argv[1]).write_bytes(p2.serialize())

class MissingReplayValidator(fixture.ExternalCompiler):
    @property
    def replay_validator(self):
        return None

try:
    qlx.compile(
        fixture.measure_product,
        pipeline=qlx.compiler.pipelines.qec(),
        device=fixture.device_for(MissingReplayValidator()),
    )
except ValueError as error:
    assert "requires an importable provider replay validator" in str(error)
else:
    raise AssertionError("unreplayable provider published a network P2 build")

class MissingArtifactValidator(fixture.ExternalCompiler):
    validate_plan_artifact = (
        qlx.compiler.QECNetworkCompiler.validate_plan_artifact
    )

try:
    qlx.compile(
        fixture.measure_product,
        pipeline=qlx.compiler.pipelines.qec(),
        device=fixture.device_for(MissingArtifactValidator()),
    )
except NotImplementedError as error:
    assert "must validate its plan artifact" in str(error)
else:
    raise AssertionError("unchecked provider artifact reached P2")
'''


_REPLAY = r'''
import json
from pathlib import Path
import sys

try:
    import _cudaq_logical_devpath
except ModuleNotFoundError:
    pass
import cudaq.logical as qlx
from cudaq.mlir import ir as mlir_ir
from cudaq.logical.compiler.build import _build_bundle_content_sha256
from cudaq.logical.qec.lattice_surgery import network_projection
from cudaq.logical.qec.lattice_surgery._codec import _digest
import external_fixture as fixture

bundle = Path(sys.argv[1]).read_bytes()


def replay_modified(module_text):
    modified = json.loads(bundle)
    modified["module"] = module_text
    modified["content_sha256"] = _build_bundle_content_sha256(modified)
    return qlx.compiler.Build.replay(
        json.dumps(modified, sort_keys=True, separators=(",", ":")).encode())


p2 = qlx.compiler.Build.replay(bundle)
compiler = fixture.ExternalCompiler()
device = fixture.device_for(compiler)
projection = network_projection(p2, device=device)
assert projection.plan.provider_key == compiler.key
p3 = qlx.compiler.project_physical(p2, device=device)
assert p3.profile == "p3"
assert "phys.measure" in p3.to_mlir()
p3_bundle = p3.serialize()
replayed_p3 = qlx.compiler.Build.replay(p3_bundle)
assert replayed_p3.profile == "p3"
assert replayed_p3.content_sha256 == p3.content_sha256
forged_p3 = json.loads(p3_bundle)
plan_claim = (
    'qlx.qec_network_plan_sha256 = '
    f'"{projection.plan.digest}"'
)
assert forged_p3["module"].count(plan_claim) == 1
forged_p3["module"] = forged_p3["module"].replace(
    plan_claim,
    'qlx.qec_network_plan_sha256 = "sha256:' + '0' * 64 + '"',
)
forged_p3["content_sha256"] = _build_bundle_content_sha256(forged_p3)
try:
    qlx.compiler.Build.replay(json.dumps(forged_p3).encode())
except ValueError as error:
    assert "network P3 replay projection commitments differ" in str(error)
else:
    raise AssertionError("forged P3 projection commitment was accepted")

compiler.reject_artifact = True
try:
    network_projection(qlx.compiler.Build.replay(bundle), device=device)
except ValueError as error:
    assert "external provider artifact differs" in str(error)
else:
    raise AssertionError("revoked provider artifact was projected")

fixture.REVOKE_REPLAY = True
try:
    qlx.compiler.Build.replay(bundle)
except ValueError as error:
    assert "external replay artifact differs" in str(error)
else:
    raise AssertionError("revoked replay artifact was accepted")
try:
    qlx.compiler.Build.replay(p3_bundle)
except ValueError as error:
    assert "external replay artifact differs" in str(error)
else:
    raise AssertionError("revoked P3 source artifact was accepted")

# Removing the validator identity cannot turn a provider-private artifact
# into one that core treats as verified without the provider.
original_module = json.loads(bundle)["module"]
validator_field = (
    'network_replay_validator = '
    '"python:external_fixture:validate_replayed_plan", '
)
assert original_module.count(validator_field) == 1
try:
    replay_modified(original_module.replace(validator_field, ""))
except ValueError as error:
    assert "requires an importable provider replay validator" in str(error)
else:
    raise AssertionError("stripped provider validator was accepted")

# A forged request, plan, artifact, metadata, and bundle digest must still
# agree with the actual retained P0 measurement, independently of the provider.
module = mlir_ir.Module.parse(original_module, mlir_ir.Context())
root = next(operation.operation for operation in module.body.operations
            if "qlx.qec_network_request" in operation.operation.attributes)
request_attribute = root.attributes["qlx.qec_network_request"]
plan_attribute = root.attributes["qlx.qec_network_plan"]
request = json.loads(request_attribute.value)
plan = json.loads(plan_attribute.value)
old_request_digest = request["digest"]
old_plan_digest = plan["digest"]
request["actions"][0]["site"]["parameters"].update(x_mask=1, z_mask=0)
request["actions"][0]["measurement"]["terms"][0]["pauli"] = "X"
request.pop("digest")
request["digest"] = _digest(request)
plan["request_sha256"] = request["digest"]
plan["artifact"]["payload"]["pauli"] = "X"
plan["artifact"]["sha256"] = _digest(plan["artifact"]["payload"])
plan.pop("digest")
plan["digest"] = _digest(plan)
new_request_attribute = mlir_ir.StringAttr.get(
    json.dumps(request, sort_keys=True, separators=(",", ":")),
    context=module.context,
)
new_plan_attribute = mlir_ir.StringAttr.get(
    json.dumps(plan, sort_keys=True, separators=(",", ":")),
    context=module.context,
)
forged_module = original_module.replace(
    f"qlx.qec_network_request = {request_attribute}",
    f"qlx.qec_network_request = {new_request_attribute}",
).replace(
    f"qlx.qec_network_plan = {plan_attribute}",
    f"qlx.qec_network_plan = {new_plan_attribute}",
).replace(
    f'network_request_sha256 = "{old_request_digest}"',
    f'network_request_sha256 = "{request["digest"]}"',
).replace(
    f'network_plan_sha256 = "{old_plan_digest}"',
    f'network_plan_sha256 = "{plan["digest"]}"',
)
assert forged_module != original_module
fixture.REVOKE_REPLAY = False
try:
    replay_modified(forged_module)
except ValueError as error:
    assert "differs from retained P0" in str(error)
else:
    raise AssertionError("forged network request was accepted")

# A self-consistently rehashed plan still has to satisfy core's temporal
# coverage rules before its provider-private artifact is accepted.
invalid_plan = json.loads(plan_attribute.value)
invalid_plan["epochs"][0]["actions"] = ["site999"]
invalid_plan.pop("digest")
invalid_plan["digest"] = _digest(invalid_plan)
invalid_plan_attribute = mlir_ir.StringAttr.get(
    json.dumps(invalid_plan, sort_keys=True, separators=(",", ":")),
    context=module.context,
)
invalid_plan_module = original_module.replace(
    f"qlx.qec_network_plan = {plan_attribute}",
    f"qlx.qec_network_plan = {invalid_plan_attribute}",
).replace(
    f'network_plan_sha256 = "{old_plan_digest}"',
    f'network_plan_sha256 = "{invalid_plan["digest"]}"',
)
try:
    replay_modified(invalid_plan_module)
except ValueError as error:
    assert "schedule every action exactly once" in str(error)
else:
    raise AssertionError("invalid network epoch was accepted")
'''


def _run_phase(code: str, *, directory: Path, bundle: Path) -> None:
    import cudaq.logical as qlx

    package_root = Path(qlx.__file__).resolve().parents[2]
    paths = (str(directory), str(package_root), os.environ.get("PYTHONPATH", ""))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    result = subprocess.run(
        [sys.executable, "-c", code, str(bundle)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_provider_compiles_and_replays_in_an_independent_process(
        tmp_path: Path) -> None:
    (tmp_path / "external_fixture.py").write_text(
        textwrap.dedent(_PROVIDER_MODULE), encoding="utf-8")
    bundle = tmp_path / "network-build.qlx"

    _run_phase(_COMPILE, directory=tmp_path, bundle=bundle)
    _run_phase(_REPLAY, directory=tmp_path, bundle=bundle)
