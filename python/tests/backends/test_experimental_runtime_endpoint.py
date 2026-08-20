# ============================================================================ #

# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for defining a CUDA-Q runtime endpoint in Python.
#
# An endpoint is a plain object implementing one or more of the protocols in
# `cudaq._experimental`. Registering it with `set_runtime_endpoint` redirects
# the *launch* step of `cudaq.sample` / `observe` / `run` into that object,
# while compilation stays with the active target.

import cudaq
import cudaq.mlir.ir as mlir
import pytest

from cudaq._experimental import set_runtime_endpoint
from cudaq._experimental.runtime_endpoint import (
    RuntimeEndpoint,
    SupportsSample,
    SupportsObserve,
    SupportsDem,
    SupportsEstimate,
)
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


@cudaq.kernel
def kernel(n_qubits: int, array: list[int]):
    qbs = cudaq.qvector(n_qubits)
    h(qbs[0])


@cudaq.kernel
def returning_kernel() -> int:
    q = cudaq.qubit()
    return 1


class DemoEndpoint(RuntimeEndpoint):
    """Records every launch and returns a canned result."""

    def __init__(self):
        self.calls = []
        self.mlir_module = None

    def sample(self, module, args, **kwargs):
        self.calls.append(("sample", repr(args), kwargs))
        self.mlir_module = module.mlir_module
        return cudaq.SampleResult({"00": kwargs["shots_count"]})

    def observe(self, module, args, **kwargs):
        self.calls.append(("observe", repr(args), kwargs))
        self.mlir_module = module.mlir_module
        return cudaq.ObserveResult(0.25, kwargs["spin_operator"],
                                   cudaq.SampleResult())

    def dem_from_kernel(self, module, args, **kwargs):
        self.calls.append(("dem_from_kernel", repr(args), kwargs))
        self.mlir_module = module.mlir_module
        return cudaq.DEMResult(dem="error(0.125) D0 L0",
                               m2d=[[0]],
                               m2o=[[0]],
                               num_detectors=1,
                               num_observables=1,
                               num_measurements=1)

    def estimate(self, module, args, **kwargs):
        self.calls.append(("estimate", repr(args), kwargs))
        self.mlir_module = module.mlir_module
        return cudaq.EstimateResult(annotations={"estimated_by": "demo"})


def test_protocol_conformance():
    endpoint = DemoEndpoint()
    assert isinstance(endpoint, SupportsSample)
    assert isinstance(endpoint, SupportsObserve)
    assert isinstance(endpoint, SupportsDem)
    assert isinstance(endpoint, SupportsEstimate)


def test_rejects_non_endpoint():

    class NotAnEndpoint:
        pass

    with pytest.raises(TypeError, match="not a valid runtime endpoint"):
        set_runtime_endpoint(NotAnEndpoint())


def test_remove_function():

    class DummyEndpoint:

        def sample(self, module, args, **kwargs):
            pass

    endpoint = DummyEndpoint()
    set_runtime_endpoint(endpoint)
    del DummyEndpoint.sample
    with pytest.raises(
            RuntimeError,
            match=
            "Expected runtime endpoint of type 'DummyEndpoint' to implement 'sample'"
    ):
        cudaq.sample(kernel, 1, [1, 2, 3], shots_count=1)


def test_sample_launch():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    assert endpoint.calls == []
    result = cudaq.sample(kernel, 1, [1, 2, 3], shots_count=12)

    assert len(endpoint.calls) == 1
    name, args, kwargs = endpoint.calls[0]
    assert name == "sample"
    # The first argument is a scalar and decodes; the vector does not.
    assert args == "KernelArgs([1, <instance of !cc.sequence<i64>>])"
    assert kwargs["shots_count"] == 12
    assert kwargs["explicit_measurements"] is False

    # The counts produced by the endpoint reach the caller.
    assert result["00"] == 12
    assert isinstance(endpoint.mlir_module, mlir.Module)


def test_observe_launch():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    result = cudaq.observe(kernel, cudaq.spin.x(0), 2, [])

    assert len(endpoint.calls) == 1
    name, args, kwargs = endpoint.calls[0]
    assert name == "observe"
    assert args == "KernelArgs([2, <instance of !cc.sequence<i64>>])"
    assert str(kwargs["spin_operator"]) == str(cudaq.spin.x(0))

    assert result.expectation() == pytest.approx(0.25)
    assert isinstance(endpoint.mlir_module, mlir.Module)


def test_dem_launch():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    noise = cudaq.NoiseModel()
    result = cudaq.dem_from_kernel(kernel,
                                   1, [1, 2, 3],
                                   noise_model=noise,
                                   decompose_errors=True)

    assert len(endpoint.calls) == 1
    name, args, kwargs = endpoint.calls[0]
    assert name == "dem_from_kernel"
    assert args == "KernelArgs([1, <instance of !cc.sequence<i64>>])"
    # The noise model arrives as the very object that was passed in.
    assert kwargs["noise_model"] is noise
    assert kwargs["decompose_errors"] is True
    assert kwargs["fold_loops"] is False
    # `dem_from_kernel` defaults this one on.
    assert kwargs["return_measurement_matrices"] is True
    assert kwargs["approximate_disjoint_errors_threshold"] == 0.0

    # The DEM produced by the endpoint reaches the caller.
    assert result.dem == "error(0.125) D0 L0"
    assert result.num_detectors == 1
    assert isinstance(endpoint.mlir_module, mlir.Module)


def test_dem_launch_without_noise_model():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    cudaq.dem_from_kernel(kernel, 1, [1, 2, 3])

    _, _, kwargs = endpoint.calls[0]
    assert kwargs["noise_model"] is None


def test_estimate_launch():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    result = cudaq.estimate(kernel, 1, [1, 2, 3])

    assert len(endpoint.calls) == 1
    name, args, kwargs = endpoint.calls[0]
    assert name == "estimate"
    assert args == "KernelArgs([1, <instance of !cc.sequence<i64>>])"
    # A default choice function is supplied even when the caller omits one.
    assert isinstance(kwargs["choice"](), bool)

    assert isinstance(result, cudaq.EstimateResult)
    assert result.resources.count() == 0
    assert result.annotations == {"estimated_by": "demo"}
    assert isinstance(endpoint.mlir_module, mlir.Module)


def test_estimate_forwards_the_choice_function():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    cudaq.estimate(kernel, 1, [1, 2, 3], choice=lambda: True)

    _, _, kwargs = endpoint.calls[0]
    assert kwargs["choice"]() is True


def test_estimate_resources_launch():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    result = cudaq.estimate_resources(kernel, 1, [1, 2, 3])

    assert len(endpoint.calls) == 1
    name, args, kwargs = endpoint.calls[0]
    assert name == "estimate"
    assert args == "KernelArgs([1, <instance of !cc.sequence<i64>>])"
    # A default choice function is supplied even when the caller omits one.
    assert isinstance(kwargs["choice"](), bool)

    # `estimate_resources` unwraps the endpoint's EstimateResult, so the caller
    # sees the Resources it carried and not the annotations.
    assert isinstance(result, cudaq.Resources)
    assert result.count() == 0
    assert isinstance(endpoint.mlir_module, mlir.Module)


def test_estimate_resources_forwards_the_choice_function():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    cudaq.estimate_resources(kernel, 1, [1, 2, 3], choice=lambda: True)

    _, _, kwargs = endpoint.calls[0]
    assert kwargs["choice"]() is True


def test_sample_twice_reuses_the_endpoint():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)

    cudaq.sample(kernel, 1, [1, 2, 3], shots_count=10)
    cudaq.sample(kernel, 2, [1, 2, 3], shots_count=10)

    assert [args for _, args, _ in endpoint.calls] == [
        "KernelArgs([1, <instance of !cc.sequence<i64>>])",
        "KernelArgs([2, <instance of !cc.sequence<i64>>])",
    ]


def test_unimplemented_policy_raises():

    class SampleOnly:

        def sample(self, module, args, **kwargs):
            return cudaq.SampleResult()

    set_runtime_endpoint(SampleOnly())
    with pytest.raises(RuntimeError, match="Unsupported policy: 'observe'"):
        cudaq.observe(kernel, cudaq.spin.x(0), 2, [])


def test_endpoint_errors_propagate():

    class Failing:

        def sample(self, module, args, **kwargs):
            raise ValueError("backend is down")

    set_runtime_endpoint(Failing())
    # The kernel invocation crosses several language boundaries on its way back
    # out, so only the message is guaranteed to survive verbatim.
    with pytest.raises((ValueError, RuntimeError), match="backend is down"):
        cudaq.sample(kernel, 1, [1, 2, 3])


def test_endpoint_wrong_sample_return_type():

    class BadSampleEndpoint:

        def sample(self, module, args, **kwargs):
            return 42

    set_runtime_endpoint(BadSampleEndpoint())
    with pytest.raises(
            TypeError,
            match=
            "Expected runtime endpoint method 'sample' to return a SampleResult, but got 'int'"
    ):
        cudaq.sample(kernel, 1, [1, 2, 3], shots_count=1)


def test_endpoint_wrong_observe_return_type():

    class BadObserveEndpoint:

        def observe(self, module, args, **kwargs):
            return cudaq.SampleResult()

    set_runtime_endpoint(BadObserveEndpoint())
    with pytest.raises(
            TypeError,
            match=
            "Expected runtime endpoint method 'observe' to return a ObserveResult, but got 'SampleResult'"
    ):
        cudaq.observe(kernel, cudaq.spin.x(0), 2, [])


@cudaq.kernel
def _state_kernel():
    q = cudaq.qubit()
    h(q)


def test_endpoint_capability_defaults():

    class SampleEndpointNoDefaults:
        """Minimal endpoint that does not explicit inherit from RuntimeEndpoint."""

        def sample(self, module, args, **kwargs):
            ...

    set_runtime_endpoint(SampleEndpointNoDefaults())

    target = cudaq.get_target()
    # default values were still set despite not existing on the endpoint
    assert target.is_remote() is False
    assert target.is_emulated() is False
    assert cudaq_runtime.isQuantumDevice() is False


def test_endpoint_capability_overrides():

    class RemoteEndpoint(RuntimeEndpoint):
        is_simulator = False
        is_remote = True
        is_emulated = True

        def sample(self, module, args, **kwargs):
            ...

    set_runtime_endpoint(RemoteEndpoint())

    target = cudaq.get_target()
    assert target.is_remote() is True
    assert target.is_emulated() is True
    assert cudaq_runtime.isQuantumDevice() is True


def test_endpoint_inherits_runtime_endpoint_defaults():

    class DefaultEndpoint(RuntimeEndpoint):

        def sample(self, module, args, **kwargs):
            return cudaq.SampleResult()

    endpoint = DefaultEndpoint()
    assert endpoint.is_simulator is True
    assert endpoint.is_remote is False
    assert endpoint.is_emulated is False

    set_runtime_endpoint(endpoint)
    target = cudaq.get_target()
    assert target.is_remote() is False
    assert target.is_emulated() is False


def test_endpoint_is_simulator_flag():

    class PhysicalEndpoint:
        is_simulator = False

        def sample(self, module, args, **kwargs):
            ...

    set_runtime_endpoint(PhysicalEndpoint())
    with pytest.raises(RuntimeError, match="physical QPU"):
        cudaq.get_state(_state_kernel)


def test_reset_target_restores_the_simulator():
    endpoint = DemoEndpoint()
    set_runtime_endpoint(endpoint)
    cudaq.sample(kernel, 1, [1, 2, 3], shots_count=10)
    assert len(endpoint.calls) == 1

    cudaq.reset_target()

    # Back on the simulator: real counts, and the endpoint is not consulted.
    result = cudaq.sample(kernel, 1, [1, 2, 3], shots_count=100)
    assert result.get_total_shots() == 100
    assert len(endpoint.calls) == 1


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
