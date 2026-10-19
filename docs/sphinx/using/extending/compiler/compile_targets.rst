Using compile targets to customize compilation
**********************************************

**Note: Support for compile targets is currently experimental and may change in the
future.**

Compile targets can be used to describe multiple stages of compilation, code
generation, as well as to infer compilation pipelines from target capabilities.
They can be defined using the `cudaq._experimental.CompileTarget` Python
class.

Constructing single-stage compile targets
=========================================

Compile targets composed of a single compilation stage can easily be constructed
using one of the following constructors:

* ``CompileTarget(ir_output_profile=profile)`` defines a target made of a single
  compilation stage that lowers the input kernel to the specified IR profile. The
  only currently supported profile is ``QuakeReferenceSemanticsProfile``.
* ``CompileTarget(custom_pipeline=lowering)`` defines a target made of a single
  compilation stage, given by the custom `lowering` pipeline. `lowering` should
  be either a ``CustomPipelineLowering`` or a MLIR pipeline string.

More complex multi-stage compile targets can be constructed using the
``CompileTarget(stages=[])`` constructor, although support for this is currently
limited. ``stages``, ``ir_output_profile`` and ``custom_pipeline`` are mutually
exclusive.

Setting compile targets with ``set_target``
===========================================

To use a compile target for all kernel launches that follow, use the
``set_target`` function with the target name and the compile target:

* Use ``cudaq.set_target("name", compile_target=target)`` to run kernels
  against a pre-defined named CUDA-Q target, but overriding the compile target.
* Use ``cudaq.set_target(CustomTarget(compile_target=target, runtime_endpoint=endpoint))``
  to run kernels using a custom compile target and a custom runtime endpoint.
  
Example
=======

The example below builds an inferred target and an explicit-pipeline target,
then compiles a kernel against each through the real ``set_target`` path. A
custom runtime endpoint is used to capture the lowered MLIR rather than would 
be executed.

.. literalinclude:: ../../../snippets/python/using/examples/compile_targets.py
   :language: python
   :lines: 9-
