CUDA-Q Circuit Simulation Backends
************************************
The simulators availible in CUDA-Q are grouped in the figure below. The following sections follow the structre of the figure and provide additional technical details and code examples fo using each circuit simulator.

.. figure:: circuitsimulators.png
   :width: 600
   :align: center

.. list-table:: Example 10x5 Table
   :header-rows: 1
   :widths: 16 25 10 10 10

   * - Simulator Name
     - Purpose
     - Processor(s)
     - Precision(s)
     - N Qubits
   * - qpp-cpu
     - Basic testing
     - CPU
     - FP32
     - < 10
   * - nvidia
     - Accelerated SV simulation
     - Single GPU
     - FP32/FP64
     - < 33 (A100)
   * - mgpu
     - Large SV simulation
     - MGMN
     - FP32/FP64
     - 33 +
   * - tensornet
     - TN simulation of shallow circuits with low entanglement
     - MGMN
     - FP32
     - Thousands 
   * - tensornet-mps
     - Matrix product state appoximated TN simulations
     - Single GPU
     - FP32
     - Thousands
   * - nvidia mqpu
     - Asynchronous distribution across multiple simulated QPUs
     - Multiple GPUs 
     - FP32/FP64
     - < 33
   * - remote mqpu
     - Combine mqpu with other backend like tensornet and mgpu
     - MGMN
     - varies
     - varies
   * - density-matrix-cpu
     - Noisy simulation
     - CPU
     - FP32
     - < 10
   * - fermioniq
     - Approximate TN simulation
     - MGMN
     - FP32
     - Thousands
   * - stim
     - QEC simulation
     - CPU
     - FP32
     - < 10



.. toctree::
   :maxdepth: 2
      
        State Vector Simulators <sims/svsims.rst>
        Tensor Network Simulators <sims/tnsims.rst>
        Multi-QPU Simulators <sims/mqpusims.rst>
        Density Matrix Simulators <sims/densitymatrix.rst>
        Other Simulators <sims/othersims.rst>

