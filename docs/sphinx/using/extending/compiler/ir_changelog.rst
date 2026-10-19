Value-semantics IR change log
*****************************

This page records changes to fully value-semantics Quake IR exposed at the
backend interface and its accompanying CC operations.

Producing this IR
=================

The ``prepare-for-wireset`` pipeline prepares supported kernels for
value-semantics Quake IR. It uses ``convert-to-linear-values`` to represent
quantum data as explicit values.

This pipeline is used, for example, by Quantum Machines and CUDA-Q Logical.

Upcoming release (``main`` branch)
==================================

Changes since 0.16.0.

Quake
-----

* Renamed ``quake.log_output`` to ``quake.evince`` to reflect its role in
  making values observable, rather than logging output. (`#5368
  <https://github.com/NVIDIA/cuda-quantum/pull/5368>`_)

* Added support for external quantum operations represented by ``quake.apply``
  calls with explicit quantum-value results. (`#5367
  <https://github.com/NVIDIA/cuda-quantum/pull/5367>`_)

* Deprecated ``quake.call_by_ref`` in favor of ``quake.apply``.
  ``convert-to-linear-values`` now emits ``quake.apply`` instead.
  ``quake.apply`` also accepts ``!quake.cable<N>`` arguments for ``!quake.struq``
  parameters when the struq's constant member sizes sum to ``N``. (`#5374
  <https://github.com/NVIDIA/cuda-quantum/pull/5374>`_)

CC
--

* Annotated ``cc.scope`` operations now print as ``cc.scope atomic { ... }``
  instead of ``cc.scope { ... } {atomic_quantum_region}``. (`#5358
  <https://github.com/NVIDIA/cuda-quantum/pull/5358>`_)

0.16.0
======

Changes since 0.15.1.

Quake
-----

* Moved ``cc.log_output`` to ``quake.log_output``. Its assembly uses a
  functional type, with explicit results that thread wire and cable values.
  (`#5113 <https://github.com/NVIDIA/cuda-quantum/pull/5113>`_)

* ``quake.apply`` assembly syntax now encloses call arguments in parentheses:
  ``quake.apply @callee(%args) : ...``. (`#4940
  <https://github.com/NVIDIA/cuda-quantum/pull/4940>`_)

CC
--

* Renamed ``!cc.stdvec`` to ``!cc.sequence`` and ``cc.stdvec_init``,
  ``cc.stdvec_data``, and ``cc.stdvec_size`` to their ``cc.sequence_*``
  equivalents. (`#5089 <https://github.com/NVIDIA/cuda-quantum/pull/5089>`_)

* Added the optional ``atomic_quantum_region`` attribute to ``cc.scope``.
  When annotated, the scope preserves an explicit optimization boundary around
  its quantum operations. (`#5128
  <https://github.com/NVIDIA/cuda-quantum/pull/5128>`_)
