CUDA-Q Dynamics
===============
.. _dynamics_examples:

This section contains examples for CUDA-Q Dynamics in both Python and C++. For a conceptual
overview of the ``evolve`` API, see the :ref:`Dynamics Simulation <dynamics>` page.

Python Examples (Jupyter Notebooks)
-------------------------------------

The notebooks below contain groups of examples using CUDA-Q Dynamics.  The first two notebooks provide an introduction to CUDA-Q Dynamics appropriate for new users.

Download the notebooks below `here <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/python/dynamics>`_.


.. toctree::
   :maxdepth: 2

      Introduction to CUDA-Q Dynamics (Jaynes-Cummings Model) <../../examples/python/dynamics/dynamics_intro_1.ipynb>
      Introduction to CUDA-Q Dynamics (Time Dependent Hamiltonians) <../../examples/python/dynamics/dynamics_intro_2.ipynb>
      Superconducting Qubits <../../examples/python/dynamics/superconducting.ipynb>
      Spin Qubits <../../examples/python/dynamics/spinqubits.ipynb>
      Trapped Ion Qubits <../../examples/python/dynamics/iontrap.ipynb>
      Control <../../examples/python/dynamics/control.ipynb>

.. |:spellcheck-disable:| replace:: \

C++ Examples
--------------

The following C++ examples demonstrate core CUDA-Q Dynamics capabilities. Each example can
be compiled and run with:

.. code:: bash

   nvq++ --target dynamics <example>.cpp -o a.out && ./a.out

The source files are available in the
`CUDA-Q repository <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/cpp/dynamics>`__.

Introduction: Single Qubit Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the basic workflow for time-evolving a single qubit under a
transverse field Hamiltonian, with and without dissipation (collapse operators).

.. literalinclude:: ../../examples/cpp/dynamics/qubit_dynamics.cpp
   :language: cpp

Introduction: Cavity QED (Jaynes-Cummings Model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates a two-level atom coupled to a single-mode optical cavity, known
as the Jaynes-Cummings model. It demonstrates how to set up composite quantum systems
with different subsystem dimensions.

.. literalinclude:: ../../examples/cpp/dynamics/cavity_qed.cpp
   :language: cpp

Superconducting Qubits: Cross-Resonance Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates the cross-resonance interaction between two coupled superconducting
qubits, a key primitive for entangling gates in superconducting hardware. It demonstrates
time-dependent Hamiltonians and batched state evolution.

.. literalinclude:: ../../examples/cpp/dynamics/cross_resonance.cpp
   :language: cpp

Spin Qubits: Heisenberg Spin Chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates the time evolution of a Heisenberg spin chain, a canonical model
for studying quantum magnetism and entanglement dynamics in spin qubit systems.

.. literalinclude:: ../../examples/cpp/dynamics/heisenberg_model.cpp
   :language: cpp

Control: Driven Qubit
~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates qubit control via a time-dependent driving Hamiltonian. It
shows how to construct schedules with named time parameters and time-dependent coefficient
callbacks for modelling control pulses.

.. literalinclude:: ../../examples/cpp/dynamics/qubit_control.cpp
   :language: cpp

State Batching
~~~~~~~~~~~~~~~

Batching multiple initial states in a single ``evolve`` call enables efficient process
tomography and parallel parameter sweeps. This example evolves four SIC-POVM states under
the same Hamiltonian simultaneously.

.. literalinclude:: ../../examples/cpp/dynamics/qubit_dynamics_batched.cpp
   :language: cpp

Numerical Integrators
~~~~~~~~~~~~~~~~~~~~~~~

CUDA-Q provides three numerical integrators for the ``dynamics`` target.

The following example shows how to use these integrators on the same single-qubit problem:

.. literalinclude:: ../../snippets/cpp/using/backends/dynamics_integrators.cpp
   :language: cpp
   :start-after: [Begin RungeKutta]
   :end-before: [End RungeKutta]

The Crank-Nicolson integrator:

.. literalinclude:: ../../snippets/cpp/using/backends/dynamics_integrators.cpp
   :language: cpp
   :start-after: [Begin CrankNicolson]
   :end-before: [End CrankNicolson]

The Magnus expansion integrator:

.. literalinclude:: ../../snippets/cpp/using/backends/dynamics_integrators.cpp
   :language: cpp
   :start-after: [Begin MagnusExpansion]
   :end-before: [End MagnusExpansion]

.. |:spellcheck-enable:| replace:: \
