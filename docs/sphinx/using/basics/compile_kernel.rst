Defining targets and customizing compilation
--------------------------------------------

.. note::

  This page is currently Python only.

  The last sections of this page, from
  :ref:`compile-kernel-custom-targets` onwards, rely on experimental APIs to
  control the compilation. These APIs are likely to change significantly in
  upcoming releases. Please do not hesitate to open an issue on
  `our GitHub repository <https://github.com/NVIDIA/cuda-quantum/issues>`__ if
  you run into problems, or would like to share suggestions or make feature requests.

Behind the scenes, CUDA-Q must compile kernels defined in Python or C++
into a form that can be executed on the target. This happens transparently
any time a quantum kernel is launched (known as just-in-time (JIT) compilation).
In this page, we look at the options available to configure and modify this
compilation step.

Compiling for built-in targets
++++++++++++++++++++++++++++++

Consider the circuit below. It is a simple 8-qubit circuit made of alternating
layers of ``rz`` and entangling ``x.ctrl`` gates. Trotter steps like these are
often used to construct circuits that simulate Hamiltonians, but there is no
need to understand this for the purposes of this page.

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin Kernel]
    :end-before: [End Kernel]

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
kernel launch functions, such as ``sample``, ``observe`` or ``estimate``, and
that controls how aggressive the optimizations that the compiler applies should
be. Higher levels enable better optimizations, which can increase the
compilation time but will result in fewer gates to be executed on the quantum
device. When no level is specified, the compiler uses level 2.

.. note::

  Advanced compiler optimizations (``optimization_level`` greater than 1) are
  currently only enabled by default for targets using the Clifford+T gate set.
  Support for optimizations on noisy hardware and simulators is still
  experimental and can be enabled using custom targets, see below.

Depending on the compile target, the compiler will prioritize different metrics
and may optimize for different cost functions. Using the previous Clifford+T
target, we can see a 75% reduction in T gate count as the optimization level
increases:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin OptimizationLevel]
    :end-before: [End OptimizationLevel]

Since level 2 is the default, its result is identical to the one obtained in
the previous section.

A Clifford+T target is also shipped with the
`cudaq-logical <../../preview/logical/index.html#http://>`__ package, CUDA-Q's
quantum error correction toolkit. It comes with a simple constructor to
override the desired approximation precision:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin Logical]
    :end-before: [End Logical]
    :dedent: 4

With the same precision as the built-in target, the T counts match the ones
above. Loosening the precision reduces the number of T gates required to
approximate each rotation, at the cost of a less accurate circuit.

.. _compile-kernel-custom-targets:

Using custom compile targets for finer-grained compilation control
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Using ``optimization_level`` to tweak the compiler is a pretty coarse setting.
You can get into more detail by tweaking the built-in targets to suit your
needs, or even defining your own custom compile targets from scratch. From here
on, we rely on the experimental ``cudaq._experimental`` module, see the note at
the top of this page.

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
``x(1)``, that is an ``x`` gate with one control), Hadamards and ``rz``
rotations. As a result, the compiler no longer synthesizes the rotations
present in the ``trotter`` kernel:

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
Hadamards and ``rx`` and ``rz`` rotations, and minimizing the number of
two-qubit gates:

.. literalinclude:: ../../snippets/python/using/compile_kernel.py
    :language: python
    :start-after: [Begin CustomTarget]
    :end-before: [End CustomTarget]

For more details on the available profiles and on how to build multi-stage
compile targets, see
:doc:`Using compile targets to customize compilation <../extending/compiler/compile_targets>`.

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

Other than for experimentation, we strongly advise users against this. The
internal CUDA-Q compiler pipelines change over time. A pipeline that works well
today may no longer do so with the next release. What's even more likely (and
worse in a way) is that your custom pipeline continues to work but silently
worsens over time. A fixed pipeline will quickly become outdated and will lock
you out of the regular performance improvements that ship with each CUDA-Q
release.
