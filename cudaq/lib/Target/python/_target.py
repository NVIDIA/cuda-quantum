# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile target types.

Plain dataclasses that mirror the corresponding C++ types. The objects in Python
are deep copies from the original C++ data. Modifying the current compile target,
would be done as follows:

```python
target = cudaq.get_compile_target()
target.stages[0].ir_output_profile.basis_gates.append("h")
cudaq.set_target("qpp-cpu", compile_target=target)
```
"""

# # Design

# We do not bind the C++ types in Python. Instead, we define custom nanobind casters
# between the Python dataclasses and the C++ types. Whenever data crosses the
# language boundary (e.g. `cudaq.set_target` or `cudaq.get_compile_target`), the
# data is deep-copied into the target language. Modifying a Python object will not
# modify the original C++ object.

# Python never converts values: after `a.foo = b`, `a.foo == b`. Values are only
# converted, or rejected, when they cross into C++. Type hints list the types that
# conversion accepts, e.g. an enum member or its name.

# # Why not bind the C++ types?

# When exposing C++ types through bindings, we must either copy the data at every property
# access, or expose the underlying data as references to C++-owned memory. References
# to C++ memory are unsafe and very hard to get right: there is no way to prevent a Python
# user to keep a pointer around long after the C++ object has been destroyed.

# The problem with copying the data at every property access is more subtle: when writing
# `foo.bar.baz = 3`, the user expects the `foo` object to have been modified. Instead,
# a nanobind binding will create a copy of `foo.bar` and assign to it. As a result, `foo`
# remains unchanged. The only way to modify `foo` would be to write:
# ```python
# foo_bar = foo.bar
# foo_bar.baz = 3
# foo.bar = foo_bar
# ```

import enum
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Literal, Optional, Union


class NoiseSupport(enum.Enum):
    """What a target does with the noise a kernel applies with `apply_noise`.
    
    - `erase`: `apply_noise` operations are erased.
    - `supported`: `apply_noise` are passed as-is to the target.
    - `unsupported`: kernels with `apply_noise` are rejected.
    """
    erase = enum.auto()
    supported = enum.auto()
    unsupported = enum.auto()


class CostModel(enum.Enum):
    """The cost function to use for gate count optimization in the compiler."""
    t_count = enum.auto()
    two_qubit_gate_count = enum.auto()


class LoopUnrolling(enum.Enum):
    """Which loops a `quake-unmapped` stage unrolls when it converts to wires."""
    all = enum.auto()
    index_use = enum.auto()


@dataclass
class UncheckedProfile:
    """IR profile that imposes no constraint."""


@dataclass
class QuakeReferenceSemanticsProfile:
    """IR profile for reference-semantics Quake."""
    basis_gates: list[str] = field(default_factory=list)
    measurement_type: Literal["any", "single_qubit"] = "any"
    reset_after_measure: bool = False


@dataclass
class QuakeUnmappedProfile:
    """IR profile for Quake before qubit mapping."""
    basis_gates: list[str] = field(default_factory=list)
    approximation_tolerance: float = 0.0
    cost_model: Union[CostModel, str, None] = None
    unroll_loops: Union[LoopUnrolling, str] = LoopUnrolling.all
    max_unroll_iterations: int = 1024


@dataclass
class QuakeMappedProfile:
    """IR profile for Quake after qubit mapping onto `device`."""
    basis_gates: list[str] = field(default_factory=list)
    device: Optional[str] = None


Profile = Union[UncheckedProfile, QuakeReferenceSemanticsProfile,
                QuakeUnmappedProfile, QuakeMappedProfile]


@dataclass
class CustomPipelineLowering:
    """An explicit MLIR pass pipeline.

    `override_pass_pipeline` replaces the whole compilation pipeline and cannot
    be combined with the `high_level_pipeline`, `mid_level_pipeline` and
    `low_level_pipeline` fragments.
    """
    override_pass_pipeline: str = ""
    high_level_pipeline: str = ""
    mid_level_pipeline: str = ""
    low_level_pipeline: str = ""

    def __post_init__(self):
        self.verify()

    def verify(self):
        if self.override_pass_pipeline and (self.high_level_pipeline or
                                            self.mid_level_pipeline or
                                            self.low_level_pipeline):
            raise ValueError("override_pass_pipeline cannot be combined with "
                             "pipeline fragments")


@dataclass
class Codegen:
    """Code generation, running after all compilation stages."""
    emission_format: str = ""
    post_codegen_passes: str = ""


@dataclass
class CompilationStage:
    """
    One compilation stage: the IR profile it lowers to, and how to get there.

    A `lowering` of None infers the pipeline from the registered handlers.
    """
    ir_output_profile: Profile
    lowering: Optional[CustomPipelineLowering] = None


@dataclass
class CompileTarget:
    """Target properties used to define the compilation pipeline.

    Instead of `stages`, a single-stage target can be given by either
    `ir_output_profile` or `custom_pipeline` (a pipeline string or a
    `CustomPipelineLowering`):

    ```python
    CompileTarget(ir_output_profile=QuakeUnmappedProfile())
    # is equivalent to
    CompileTarget(stages=[
        CompilationStage(ir_output_profile=QuakeUnmappedProfile()),
    ])

    CompileTarget(custom_pipeline="canonicalize,cse")
    # is equivalent to
    CompileTarget(stages=[
        CompilationStage(
            ir_output_profile=UncheckedProfile(),
            lowering=CustomPipelineLowering(
                override_pass_pipeline="canonicalize,cse"),
        ),
    ])
    ```
    """

    stages: list[CompilationStage] = field(default_factory=list)
    codegen: Union[Codegen, str] = field(default_factory=Codegen)
    override_aot_compilation: bool = False
    support_conditionals_on_measure_results: bool = True
    support_device_calls: bool = False
    support_resource_counts: bool = True
    support_explicit_measurements: bool = True
    support_observable_measurements: bool = True
    support_sample_without_measurements: bool = True
    fully_specialize: bool = True
    preserve_loops: bool = False
    noise_support: Union[NoiseSupport, str] = NoiseSupport.erase
    erase_qec: bool = True
    argument_synth_change_semantics: bool = True
    disable_qubit_mapping: bool = False
    replace_state_with_kernel: bool = False
    reset_on_random_seed_set: bool = False

    # These are aliases for some common options - not stored on the dataclass,
    # but `__init__` accepts them and passes them to `__post_init__`.
    _: KW_ONLY
    ir_output_profile: InitVar[Optional[Profile]] = None
    custom_pipeline: InitVar[Union[CustomPipelineLowering, str, None]] = None

    def __post_init__(self, ir_output_profile, custom_pipeline):

        def set_stages(stage, shortcut):
            if self.stages:
                raise TypeError(
                    f"'{shortcut}' cannot be combined with a non-empty 'stages' value"
                )
            self.stages = [stage]

        if ir_output_profile is not None and custom_pipeline is not None:
            raise TypeError(
                "'ir_output_profile' and 'custom_pipeline' are mutually exclusive"
            )
        if ir_output_profile is not None:
            set_stages(CompilationStage(ir_output_profile), "ir_output_profile")
        elif custom_pipeline is not None:
            if isinstance(custom_pipeline, str):
                custom_pipeline = CustomPipelineLowering(custom_pipeline)
            set_stages(CompilationStage(UncheckedProfile(), custom_pipeline),
                       "custom_pipeline")
