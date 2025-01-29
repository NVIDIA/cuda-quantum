CUDA-Q Circuit Simulation Backends
************************************
The simulators available in CUDA-Q are grouped in the figure below. The following sections follow the structre of the figure and provide additional technical details and code examples fo using each circuit simulator.

.. figure:: circuitsimulators.png
   :width: 600
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
   * - qpp-cpu
     - State Vector
     - Testing and small applications
     - CPU
     - FP32
     - < 28
   * - nvidia
     - State Vector
     - General purpose (default)
     - Single GPU
     - FP32/FP64
     - < 33/32 (64 GB)
   * - mgpu
     - State Vector
     - Large-scale simulation
     - multi-GPU multi-node
     - FP32/FP64
     - 33 +
   * - tensornet
     - Tensor Network
     - Shallow-depth (low-entanglement) and high width circuits
     - multi-GPU multi-node
     - FP32/FP64
     - Thousands 
   * - tensornet-mps
     - Matrix Product State
     - Square-shaped circuits
     - Single GPU
     - FP32/FP64
     - Hundreds
   * - fermioniq
     - Various
     - Various
     - Single GPU
     - Various
     - Various
   * - nvidia mqpu
     - State Vector 
     - Asynchronous distribution across multiple simulated QPUs to speedup applications
     - multi-GPU multi-node
     - FP32/FP64
     - < 33/32 (64 GB)
   * - remote mqpu
     - SV/TN
     - Combine mqpu with other backend like tensornet and mgpu
     - varies
     - varies
     - varies
   * - density-matrix-cpu
     - Density Matrix
     - Noisy simulations
     - CPU
     - FP32
     - <14
   * - stim
     - Stabilizer 
     - QEC simulation
     - CPU
     - N/A
     - Thousands +
   * - orca-photonics
     - State Vector
     - Photonics
     - CPU
     - FP64
     - Varies on qudit level



.. toctree::
   :maxdepth: 2
      
        State Vector Simulators <sims/svsims.rst>
        Tensor Network Simulators <sims/tnsims.rst>
        Multi-QPU Simulators <sims/mqpusims.rst>
        Noisy Simulators <sims/noisy.rst>
        Photonics Simulators <sims/photonics.rst>

