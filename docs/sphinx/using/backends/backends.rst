CUDA-Q Backends
**********************

.. toctree::
   :caption: Backend Targets
   :maxdepth: 1

      Simulation <simulators.rst>
      Quantum Hardware <hardware.rst>
      NVIDIA Quantum Cloud <nvqc.rst>
      Multi-Processor Platforms <platform.rst>

**The following is a comprehensive list of the available targets in CUDA-Q:**

* :ref:`braket <braket-backend>`
* :ref:`density-matrix-cpu <default-simulator>`
* :ref:`fermioniq <fermioniq-backend>`
* :ref:`ionq <ionq-backend>`
* :ref:`anyon <anyon-backend>`
* :ref:`iqm <iqm-backend>`
* :ref:`nvidia <nvidia-backend>`
* :ref:`nvidia-fp64 <nvidia-fp64-backend>`
* :ref:`nvidia-mgpu <nvidia-mgpu-backend>`
* :ref:`nvidia-mqpu <mqpu-platform>`
* :ref:`nvidia-mqpu-fp64 <mqpu-platform>`
* :doc:`nvqc <nvqc>`
* :ref:`oqc <oqc-backend>`
* :ref:`orca <orca-backend>`
* :ref:`qpp-cpu <qpp-cpu-backend>`
* :ref:`quantinuum <quantinuum-backend>`
* :ref:`quera <quera-backend>`
* :ref:`remote-mqpu <mqpu-platform>`
* :ref:`stim <stim-backend>`
* :ref:`tensornet <tensor-backends>`
* :ref:`tensornet-mps <tensor-backends>`

.. deprecated:: 0.8
   The `nvidia-fp64`, `nvidia-mgpu`, `nvidia-mqpu`, and `nvidia-mqpu-fp64` targets can be 
   enabled as extensions of the unified `nvidia` target (see `nvidia` :ref:`target documentation <nvidia-backend>`).
   These target names might be removed in a future release.