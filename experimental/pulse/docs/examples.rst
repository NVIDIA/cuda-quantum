.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Examples
========

cudaq-pulse includes a comprehensive set of examples demonstrating
pulse-level quantum programming, from single-qubit Rabi oscillations
to multi-qubit QEC circuits.

All examples are in the ``examples/`` directory and use the canonical
import convention:

.. code-block:: python

   import cudaq_pulse as pulse

Core Examples
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - #
     - File
     - Description
   * - 01
     - ``01_single_qubit_rabi.py``
     - Single-qubit Rabi oscillation with a square pulse of varying amplitude
   * - 02
     - ``02_two_qubit_cross_resonance.py``
     - Two-qubit CNOT via echoed cross-resonance driving
   * - 03
     - ``03_t1_t2_dissipator.py``
     - T1/T2 measurement with Lindblad dissipator modeling
   * - 04
     - ``04_echo_paper.py``
     - Canonical Hahn spin-echo sequence (pi/2 - tau - pi - tau - pi/2)
   * - 05
     - ``05_waveform_gallery.py``
     - All 8 waveform constructors and 5 algebraic combinators
   * - 06
     - ``06_phase_frequency_control.py``
     - Phase and frequency manipulation primitives on tone channels
   * - 07
     - ``07_multi_qubit_sync.py``
     - Multi-qubit programs with sync and ``pulse.qvec_ref()`` allocation
   * - 08
     - ``08_readout_and_branching.py``
     - Readout channels and measurement-conditioned branching
   * - 09
     - ``09_compilation_pipeline.py``
     - Full ``pulse.compile()`` API walkthrough with pass customization
   * - 10
     - ``10_scheduling_comparison.py``
     - Side-by-side ASAP, ALAP, and RCP scheduling on a 4-qubit program
   * - 11
     - ``11_loop_optimizations.py``
     - LICM and loop strength reduction on pulse loop bodies
   * - 12
     - ``12_error_detection.py``
     - Verification error detection with intentionally malformed programs
   * - 13
     - ``13_dynamical_decoupling.py``
     - XY4, Uhrig, and CPMG dynamical decoupling sequences
   * - 14
     - ``14_randomized_benchmarking.py``
     - Single-qubit randomized benchmarking with Clifford decomposition
   * - 15
     - ``15_pulse_to_operator.py``
     - Full pulse-to-operator lowering and cuDensityMat simulation pipeline
   * - 16
     - ``16_visualization.py``
     - Pulse schedule visualization with matplotlib Gantt charts
   * - 17
     - ``17_ghz_state_prep.py``
     - N-qubit GHZ state preparation using echoed cross-resonance CNOTs

Hardware Target Examples
------------------------

**Transmon** (``examples/transmon/``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``krinner_rabi.py``
     - Rabi oscillation on the Krinner 17-qubit transmon target
   * - ``krinner_full_pipeline.py``
     - Full compilation pipeline on the Krinner target
   * - ``krinner_evolve_gpu.py``
     - GPU-accelerated evolution with the Krinner target
   * - ``krinner_surface_code_cycle.py``
     - Surface code stabilizer cycle on the Krinner lattice

**Neutral Atom** (``examples/neutral_atom/``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``rydberg_blockade_demo.py``
     - Two-atom Rydberg blockade interaction
   * - ``rydberg_chain_adiabatic.py``
     - Adiabatic sweep on a 1D Rydberg atom chain

Running Examples
----------------

.. code-block:: bash

   cd cudaq-pulse
   PYTHONPATH=core/frontend:build/core/mlir/bindings python examples/01_single_qubit_rabi.py
