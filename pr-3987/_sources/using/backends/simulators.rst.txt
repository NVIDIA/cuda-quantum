CUDA-Q Circuit Simulation Backends
************************************
.. _simulators:

The simulators available in CUDA-Q are grouped in the figure below. The 
following sections follow the structure of the figure and provide additional 
technical details and code examples for using each circuit simulator.

.. figure:: circuitsimulators.png
   :width: 900
   :align: center

.. list-table:: Simulators In CUDA-Q
   :header-rows: 1
   :widths: 20 20 25 10 10 16

   * - Simulator Name
     - Method
     - Purpose
     - Processor(s)
     - Precision(s)
     - N Qubits
   * - `qpp-cpu`
     - State Vector
     - Testing and small applications
     - CPU
     - double
     - < 28
   * - `nvidia` *
     - State Vector
     - General purpose (default); Trajectory simulation for noisy circuits
     - Single GPU
     - single (default) / double
     - < 33 / 32 (64 GB)
   * - `nvidia, option=mgpu` *
     - State Vector
     - Large-scale simulation
     - multi-GPU multi-node
     - single (default) / double
     - 33+
   * - `tensornet` *
     - Tensor Network
     - Shallow-depth (low-entanglement) and high width circuits (exact)
     - multi-GPU multi-node
     - double (default) / single
     - Thousands 
   * - `tensornet-mps` *
     - Matrix Product State
     - Square-shaped circuits (approximate)
     - Single GPU
     - double (default) / single
     - Hundreds
   * - `fermioniq`
     - Matrix Product State
     - Square-shaped circuits (approximate)
     - Single GPU
     - double
     - Hundreds
   * - `nvidia, option=mqpu` *
     - State Vector 
     - Asynchronous distribution across multiple simulated QPUs to speedup applications
     - multi-GPU multi-node
     - single (default) / double
     - < 33 / 32 (64 GB)
   * - `remote-mqpu` *
     - State Vector / Tensor Network
     - Combine `mqpu` with other backend like `tensornet` and `mgpu`
     - varies
     - varies
     - varies
   * - Trajectory Noisy Simulation
     - works with all simulators marked *
     - Noisy trajectory simulations
     - multi-GPU multi-node
     - double
     - varies
   * - `density-matrix-cpu`
     - Density Matrix
     - Noisy simulations
     - CPU
     - double
     - < 14
   * - `stim`
     - Stabilizer 
     - QEC simulation
     - CPU
     - N/A
     - Thousands +
   * - `orca-photonics`
     - State Vector
     - Photonics
     - CPU
     - double
     - Varies on qudit level



.. toctree::
   :maxdepth: 2
      
        State Vector Simulators <sims/svsims.rst>
        Tensor Network Simulators <sims/tnsims.rst>
        Multi-QPU Simulators <sims/mqpusims.rst>
        Noisy Simulators <sims/noisy.rst>
        Photonics Simulators <sims/photonics.rst>

