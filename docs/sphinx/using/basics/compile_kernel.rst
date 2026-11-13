Defining targets and customizing compilation
--------------------------------------------

.. note::

  This page is currently Python only.

  The last sections of this page, from
  :ref:`compile-kernel-custom-targets` onward, rely on experimental APIs to
  control the compilation. These APIs are likely to change significantly in
  upcoming releases. Please do not hesitate to open an issue on
  `our GitHub repository <https://github.com/NVIDIA/cuda-quantum/issues>`__ if
  you run into problems, or would like to share suggestions or make feature requests.

Behind the scenes, CUDA-Q must compile kernels defined in Python or C++
into a form that can be executed on the target. This happens transparently
any time a quantum kernel is launched (known as just-in-time (JIT) compilation).
In this page, we look at the options available to configure and modify this
compilation step, focusing on the most common use cases. For a complete
description of compile targets, including IR profiles, multi-stage targets and
how compilation pipelines are inferred, see
:doc:`Using compile targets to customize compilation <../extending/compiler/compile_targets>`.

Compiling for built-in targets
++++++++++++++++++++++++++++++

Throughout this page, we compile a kernel ``qpe`` that implements quantum
phase estimation on 6 qubits, with 4 data qubits and 2 ancilla. Have a look at
the :download:`source <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/snippets/python/using/compile_kernel.py>`.
if you are interested in the full implementation.

At the most basic level, a compiler's job is to take this high-level
description of a computation and produce a circuit that a given target is able
to execute. Setting for instance the target to ``iqm`` will cause the compiler
to produce a circuit that uses the basis gates available on IQM's hardware. In
emulation mode, the compiled kernel is not submitted to IQM's servers. To
inspect the compiled circuit, we use :func:`cudaq.estimate`, which counts the
gates and qubits that the compiled kernel uses:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin IQM]
    :end-before: [End IQM]

As you can see from the resource estimates, the resulting circuit only contains
``cz`` and ``phased_rx`` gates, alongside measurements.

.. note::

  When running the code above, CUDA-Q will warn you that it is unable to
  obtain a copy of the device architecture, which it would normally use for
  routing the circuit. You can fix the warning by either providing a valid IQM
  token or passing a ``mapping_file`` directly as an option to
  ``cudaq.set_target``. See the :ref:`IQM backend <iqm-backend>` and its
  :doc:`advanced use cases <../backends/hardware/backend_iqm>` for more
  information.

The compiler also supports synthesizing circuits to discrete gate sets, often
required to encode circuits using quantum error correcting codes. We can for
instance select CUDA-Q's built-in Clifford+T target to express the same kernel
in terms of Cliffords (CNOT, H, S, X, Z) and T gates:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin CliffordT]
    :end-before: [End CliffordT]

Note that as this gate set is discrete, the resulting circuit can only
implement arbitrary rotation angles up to a certain approximation error (by
default ``1e-10``). The cudaq-logical package allows you to configure the
precision of the approximation, see below. For more details on how rotations
are approximated, see :doc:`Rotation Synthesis <../examples/rotation_synthesis>`.

Controlling the optimization level of the compiler
++++++++++++++++++++++++++++++++++++++++++++++++++

The simplest configuration option that can be modified to adjust the behavior
of the compiler is the optimization level. This is a number between 0 and 3
that can be passed as the ``optimization_level`` argument to any of the CUDA-Q
kernel launch functions, such as ``sample``, ``observe`` or ``estimate``. It
controls which compiler optimizations should be enabled.  As with the
``-O0`` to ``-O3`` flags of classical compilers, higher levels correspond to
more aggressive optimizations, which should result in fewer gates in the final
compiled program:

- **Level 0**: only the transformations required to lower the kernel to the
  target, such as decomposing gates into the target's gate set.
- **Level 1**: cheap local rewrites of adjacent or commuting gates, such as gate
  cancellation and rotation merging.
- **Level 2**: analyses across the whole kernel, such as phase folding and
  re-synthesis. These take more compile time but usually produce smaller
  circuits.
- **Level 3**: reserved for more expensive search-based optimizations. It
  currently behaves like level 2.

When no level is specified, the compiler currently uses level 2.

.. note::

  Advanced compiler optimizations (``optimization_level`` greater than 1) are
  currently only enabled by default for targets using the Clifford+T gate set.
  Support for optimizations on noisy hardware and simulators is still
  experimental and can be enabled using custom targets, see below.

Depending on the compile target, the compiler will prioritize different metrics
and may optimize for different cost functions. Using the previous Clifford+T
target, we can see a 67% reduction in T gate count as the optimization level
increases:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin OptimizationLevel]
    :end-before: [End OptimizationLevel]

Since level 2 is the default, its result is identical to the one obtained in
the previous section. Optimization levels 2 and 3 are currently identical.

A Clifford+T target is also shipped with the
`cudaq-logical <../../preview/logical/index.html#http://>`__ package, CUDA-Q's
quantum error correction toolkit. It comes with a simple constructor to
override the desired approximation precision:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin Logical]
    :end-before: [End Logical]
    :dedent: 4

Loosening the precision reduces the number of T gates required to
approximate each rotation, at the cost of a less accurate circuit.

.. _compile-kernel-custom-targets:

Using custom compile targets for finer-grained compilation control
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The optimization level is a useful high-level setting to control how much
effort is spent on optimization. When finer-grained control is required over
the compilation, you can tweak the built-in targets to suit your needs, or
even define your own custom compile targets from scratch. From here on, we rely
on the experimental ``cudaq._experimental`` module, see the note at the top of
this page. The examples below cover a few common use cases;
see :doc:`Using compile targets to customize compilation <../extending/compiler/compile_targets>`
for the other available profiles, multi-stage targets and lowering options.

To start with, here is how to retrieve the full configuration of the built-in
Clifford+T compile target we have been using:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin GetCompileTarget]
    :end-before: [End GetCompileTarget]

The Clifford+T compile target is defined by a single compilation stage
``clifford_t_target.stages[0]``. The stage is defined using an
``ir_output_profile``, which characterizes the kinds of quantum circuits that
the target supports. Among other things, it lists the basis gates that the
compiled circuit may use, the tolerance used to approximate rotations, and the
cost model that the optimizations aim to minimize.

We can for instance replace the allowed basis gates with CNOTs (written
``x(1)``, that is an ``x`` gate with one control), Hadamards, ``x`` gates
and ``rz`` rotations. As a result, the compiler no longer synthesizes the rotations
present in the ``qpe`` kernel:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin ModifyTarget]
    :end-before: [End ModifyTarget]

Note that a CUDA-Q target is made of a runtime component (*where* a kernel
should execute), and a compile component (*how* a kernel should be compiled).
We only wish to override the compile part of the target. This is why when using
``cudaq.set_target``, we still specify a base built-in target (in our case, the
QPP simulator): the runtime component remains unchanged from the built-in,
while the compile target gets overridden by our custom target.

However, as configured, our custom target still optimizes for T count. This no
longer makes sense now that we have changed the gate set. We can instead define
our own compile target from scratch, using controlled-Z gates (written ``z(1)``),
Hadamards, ``x`` gates and ``rx`` and ``rz`` rotations, and minimizing the number of
two-qubit gates:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin CustomTarget]
    :end-before: [End CustomTarget]

In this case, optimization level 2 matches the performance achieved by the
T-count optimized compilation.

Customizing all the way
+++++++++++++++++++++++

Finally, to get maximal control over the compiler, users can choose to bypass
the CUDA-Q pre-defined compilation options altogether and provide a custom MLIR
compilation pipeline. The example below only decomposes the gates into the
requested basis and applies no optimizations, which is why it produces the same
circuit as the previous target at optimization level 0:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin CustomPipeline]
    :end-before: [End CustomPipeline]

Here we only override the mid-level pipeline. ``CustomPipelineLowering`` also
accepts further arguments, see
:doc:`Using compile targets to customize compilation <../extending/compiler/compile_targets>`
for more details.

Defining custom pipelines is mostly useful for experimentation, when doing research
or developing new custom targets. Expect to have to update and adjust your custom
pipeline over time, as the CUDA-Q compiler evolves. A pipeline that works well
today may become outdated and may not be designed to leverage features that will
ship with future releases of CUDA-Q.
